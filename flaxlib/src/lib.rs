// Copyright 2024 The Flax Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    ffi::PyList_Type,
    prelude::*,
    types::{IntoPyDict, PyList, PyNone, PyTuple},
    PyTypeCheck,
};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

struct FlattenCtx<'py> {
    py: Python<'py>,
    graph: Bound<'py, PyModule>,
    jax_array_type: Bound<'py, PyAny>,
    np_array_type: Bound<'py, PyAny>,
    variable_type: Bound<'py, PyAny>,
    node_def_type: Bound<'py, PyAny>,
}

// def _graph_flatten(
//     path: PathParts,
//     ref_index: RefMap[tp.Any, Index],
//     flat_state: list[tuple[PathParts, StateLeaf]],
//     node: Node,
//   ) -> NodeDef[Node] | NodeRef:
#[pyfunction]
fn flatten<'py>(
    node: &Bound<'py, PyAny>,
    ref_index: &Bound<'py, PyAny>,
) -> PyResult<(
    Bound<'py, PyAny>,
    Vec<(Bound<'py, PyTuple>, Bound<'py, PyAny>)>,
)> {
    let py = node.py();
    let graph = py.import_bound("flax.nnx.graph")?;
    let path: Vec<Bound<PyAny>> = Vec::new();
    let mut flat_state: Vec<(Bound<PyTuple>, Bound<PyAny>)> = Vec::new();
    let variable_type = graph.getattr("Variable")?;
    let node_def_type = graph.getattr("NodeDef")?;
    let ctx = FlattenCtx::<'py> {
        py,
        graph,
        jax_array_type: py.import_bound("jax")?.getattr("Array")?,
        np_array_type: py.import_bound("numpy")?.getattr("ndarray")?,
        variable_type: variable_type,
        node_def_type: node_def_type,
    };
    let node_ref = _graph_flatten(&ctx, path, ref_index, &mut flat_state, node)?;
    return Ok((node_ref, flat_state));
}

fn _graph_flatten<'py>(
    ctx: &FlattenCtx<'py>,
    path: Vec<Bound<'py, PyAny>>,
    ref_index: &Bound<'py, PyAny>,
    flat_state: &mut Vec<(Bound<'py, PyTuple>, Bound<'py, PyAny>)>,
    node: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // if not is_node(node):
    //   raise RuntimeError(f'Unsupported type: {type(node)}, this is a bug.')
    let is_node = ctx.graph.call_method1("is_node", (node,))?.is_truthy()?;
    if !is_node {
        return Err(PyRuntimeError::new_err(format!(
            "Unsupported type: {:?}, this is a bug.",
            node.get_type()
        )));
    }

    // if node in ref_index:
    //   return NodeRef(type(node), ref_index[node])
    let node_in_ref_index = ref_index.contains(node)?;
    if node_in_ref_index {
        let node_type = node.get_type();
        let index = ref_index.get_item(node)?;
        let node_ref = ctx.graph.getattr("NodeRef")?.call1((node_type, index))?;
        return Ok(node_ref);
    }

    // node_impl = get_node_impl(node)
    let node_impl = ctx.graph.getattr("get_node_impl")?.call1((node,))?;

    // # only cache graph nodes
    // if isinstance(node_impl, GraphNodeImpl):
    //   index = len(ref_index)
    //   ref_index[node] = index
    // else:
    //   index = -1
    let index: i32;
    let is_graph_node_impl = node_impl.is_instance(&ctx.graph.getattr("GraphNodeImpl")?)?;
    if is_graph_node_impl {
        index = ref_index.len()? as i32;
        ref_index.set_item(node, index)?;
    } else {
        index = -1;
    }

    // subgraphs: list[tuple[Key, NodeDef[Node] | NodeRef]] = []
    // static_fields: list[tuple[Key, tp.Any]] = []
    // leaves: list[tuple[Key, NodeRef | None]] = []
    let mut subgraphs: Vec<(Bound<PyAny>, Bound<PyAny>)> = Vec::new();
    let mut static_fields: Vec<(Bound<PyAny>, Bound<PyAny>)> = Vec::new();
    let mut leaves: Vec<(Bound<PyAny>, Bound<PyAny>)> = Vec::new();
    let mut attributes: Vec<Bound<PyAny>> = Vec::new();

    //     values, metadata = node_impl.flatten(node)
    let values = node_impl.call_method1("flatten", (node,))?;
    let metadata = values.get_item(1)?;
    let values = values.get_item(0)?;

    // if PyList::type_check(&values) {
    let values: Vec<(Bound<PyAny>, Bound<PyAny>)> = values.extract()?;
    // }

    //     for key, value in values:
    for (key, value) in values {
        // let item = item?;
        // let key = item.get_item(0)?;
        // let value = item.get_item(1)?;

        attributes.push(key.clone());
        // child_path = (*path, key)
        let mut child_path = path.clone();
        child_path.push(key.clone());

        // if is_node(value):
        if ctx
            .graph
            .call_method1("is_node", (value.clone(),))?
            .is_truthy()?
        {
            // nodedef = _graph_flatten(child_path, ref_index, flat_state, value)
            // subgraphs.append((key, nodedef))
            let nodedef = _graph_flatten(ctx, child_path, ref_index, flat_state, &value)?;
            subgraphs.push((key.clone(), nodedef));
        //       elif isinstance(value, Variable):
        } else if value.is_instance(&ctx.variable_type)? {
            // if value in ref_index:
            if ref_index.contains(value.clone())? {
                // leaves.append((key, NodeRef(type(value), ref_index[value])))
                let variable_index = ref_index.get_item(value.clone())?;
                let node_ref = ctx
                    .graph
                    .call_method1("NodeRef", (value.get_type(), variable_index))?;
                leaves.push((key.clone(), node_ref));
            // else:
            } else {
                // flat_state.append((child_path, value.to_state()))
                // variable_index = ref_index[value] = len(ref_index)
                // leaves.append((key, NodeRef(type(value), variable_index)))
                let variable_state = value.call_method0("to_state")?;
                let child_path = PyTuple::new_bound(ctx.py, child_path);
                flat_state.push((child_path, variable_state));
                let variable_index = ref_index.len()? as i32;
                ref_index.set_item(value.clone(), variable_index)?;
                let node_ref = ctx
                    .graph
                    .call_method1("NodeRef", (value.get_type(), variable_index))?;
                leaves.push((key.clone(), node_ref));
            }
        // elif is_state_leaf(value):
        } else if ctx
            .graph
            .call_method1("is_state_leaf", (value.clone(),))?
            .is_truthy()?
        {
            // flat_state.append((child_path, value))
            // leaves.append((key, None))
            let child_path = PyTuple::new_bound(ctx.py, child_path);
            flat_state.push((child_path, value));
            leaves.push((key.clone(), ctx.py.None().into_bound(ctx.py)));
        // else:
        } else {
            //  if isinstance(value, (jax.Array, np.ndarray)):
            if value.is_instance(&ctx.jax_array_type)? || value.is_instance(&ctx.np_array_type)? {
                // path_str = '/'.join(map(str, child_path))
                // raise ValueError(
                //     f'Arrays leaves are not supported, at {path_str!r}: {value}'
                // )
                // let path_str = child_path.join("/");
                return Err(PyValueError::new_err(format!(
                    "Arrays leaves are not supported, at {:?}: {:?}",
                    child_path, value
                )));
            }
            //         static_fields.append((key, value))
            static_fields.push((key.clone(), value));
        }
    }
    // nodedef = NodeDef.create(
    //   type=node_impl.type,
    //   index=index,
    //   attributes=tuple(key for key, _ in values),
    //   subgraphs=subgraphs,
    //   static_fields=static_fields,
    //   leaves=leaves,
    //   metadata=metadata,
    //   index_mapping=None,
    // )
    let nodedef = ctx.node_def_type.call_method1(
        "create",
        (
            node_impl.getattr("type")?,
            index,
            PyTuple::new_bound(ctx.py, attributes),
            subgraphs,
            static_fields,
            leaves.into_py_dict_bound(ctx.py),
            metadata,
            PyNone::get_bound(ctx.py),
        ),
    )?;
    // return nodedef
    return Ok(nodedef);
}

/// A Python module implemented in Rust.
#[pymodule]
fn flaxlib(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {

    use pyo3::{
        types::{IntoPyDict, PyAnyMethods, PyList, PyTuple},
        PyResult, Python,
    };

    use crate::flatten;

    #[test]
    fn test_flatten() {
        pyo3::prepare_freethreaded_python();
        fn test<'py>(py: Python<'py>) -> PyResult<()> {
            let nnx = py.import_bound("flax.nnx")?;
            let graph = py.import_bound("flax.nnx.graph")?;

            let rngs = nnx.call_method1("Rngs", (0,))?;
            let kwargs = [("rngs", rngs)].into_py_dict_bound(py);
            let model = nnx.call_method("Linear", (2, 3), Some(&kwargs))?;

            let ref_map = graph.call_method0("RefMap")?;

            let (graph_def, flat_state) = flatten(&model, &ref_map)?;

            println!("graphdef rust: {:?}\n", graph_def);
            println!("flat_state rust: {:?}", flat_state);

            let path = PyTuple::empty_bound(py);
            let ref_map = graph.call_method0("RefMap")?;
            let flat_state_python = PyList::empty_bound(py);
            let graphdef_python = graph.call_method1(
                "_graph_flatten",
                (path, ref_map, &flat_state_python, &model),
            )?;

            println!("graphdef python: {:?}", graphdef_python);
            println!("flat_state python: {:?}", flat_state_python);

            Ok(())
        }
        match Python::with_gil(|py| test(py)) {
            Ok(_) => (),
            Err(e) => panic!("{:?}", e.to_string()),
        }
    }
}
