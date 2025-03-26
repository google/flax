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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/optional.h>
#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include <Python.h>

namespace nb = nanobind;
using namespace nb::literals;

// -----------------------------------
// helper functions
// -----------------------------------
intptr_t nb_id(const nb::object &obj)
{
  // Get the object ID
  return reinterpret_cast<intptr_t>(obj.ptr());
}

nb::tuple vector_to_tuple(const std::vector<nb::object> &vec)
{

  if (vec.empty())
  {
    return nb::tuple();
  }
  else
  {
    return nb::tuple(nb::cast(vec));
  }
}

// 1. Hash function for nb::object
struct NbObjectHash
{
  std::size_t operator()(const nb::object &obj) const
  {
    return nb::hash(obj);
  }
};

// 2. Equality function for nb::object (Important!)
struct NbObjectEqual
{
  bool operator()(const nb::object &a, const nb::object &b) const
  {
    return a.equal(b);
  }
};

namespace flaxlib
{
  //---------------------------------------------------------------
  // NNXContext
  //---------------------------------------------------------------

  struct PythonContext
  {
    nb::object nnx;
    nb::object graph;
    nb::object jax;
    nb::object np;
    nb::object jax_Array;
    nb::object np_ndarray;
    nb::type_object GraphNodeImpl;
    nb::type_object PytreeNodeImpl;
    nb::type_object Object;
    nb::type_object Variable;
    nb::object get_node_impl;

    PythonContext()
    {
      nnx = nb::module_::import_("flax.nnx");
      graph = nb::module_::import_("flax.nnx.graph");
      jax = nb::module_::import_("jax");
      np = nb::module_::import_("numpy");
      jax_Array = jax.attr("Array");
      np_ndarray = np.attr("ndarray");
      GraphNodeImpl = graph.attr("GraphNodeImpl");
      PytreeNodeImpl = graph.attr("PytreeNodeImpl");
      Object = nnx.attr("Object");
      Variable = graph.attr("Variable");
      get_node_impl = graph.attr("get_node_impl");
    }

    ~PythonContext()
    {
      graph.release();
      jax.release();
      np.release();
      jax_Array.release();
      np_ndarray.release();
      GraphNodeImpl.release();
      PytreeNodeImpl.release();
      Variable.release();
      get_node_impl.release();
    }
  };

  static std::optional<PythonContext> _python_context;

  PythonContext &get_python_context()
  {
    if (!_python_context)
    {
      _python_context.emplace();
    }
    return *_python_context;
  }

  //---------------------------------------------------------------
  // PytreeNodeImpl
  //---------------------------------------------------------------
  // @dataclasses.dataclass(frozen=True, slots=True)
  // class NodeImplBase(tp.Generic[Node, Leaf, AuxData]):
  //   type: type[Node]
  //   flatten: tp.Callable[[Node], tuple[tp.Sequence[tuple[Key, Leaf]], AuxData]]

  //   def node_dict(self, node: Node) -> dict[Key, Leaf]:
  //     nodes, _ = self.flatten(node)
  //     return dict(nodes)

  struct NodeImplBase
  {
    nb::object type;
    nb::object flatten;

    NodeImplBase(nb::object type, nb::object flatten)
        : type(type), flatten(flatten) {}

    nb::dict node_dict(nb::object &node)
    {
      nb::object nodes_aux = flatten(node);
      nb::dict out;
      for (auto elem : nodes_aux[0])
      {
        out[elem[0]] = elem[1];
      }
      return out;
    }

    bool __eq__(const nb::object &other_obj) const
    {
      if (!nb::isinstance<NodeImplBase>(other_obj))
      {
        return false;
      }
      NodeImplBase other = nb::cast<NodeImplBase>(other_obj);
      return type.equal(other.type) && flatten.equal(other.flatten);
    }

    int __hash__() const
    {
      return nb::hash(nb::make_tuple(type, flatten));
    }

    nb::tuple __getstate__() const
    {
      return nb::make_tuple(type, flatten);
    }

    static void __setstate__(NodeImplBase &nodeimplbase, nb::tuple &t)
    {
      new (&nodeimplbase) NodeImplBase(t[0], t[1]);
    }
  };

  // @dataclasses.dataclass(frozen=True, slots=True)
  // class GraphNodeImpl(NodeImplBase):
  //   set_key: tp.Callable[[tp.Any, Key, tp.Any], None]
  //   pop_key: tp.Callable[[tp.Any, Key], tp.Any]
  //   create_empty: tp.Callable[[tp.Any], tp.Any]
  //   clear: tp.Callable[[tp.Any], None]
  //   init: tp.Callable[[tp.Any, tp.Iterable[tuple[Key, tp.Any]]], None]

  struct GraphNodeImpl : public NodeImplBase
  {
    nb::object set_key;
    nb::object pop_key;
    nb::object create_empty;
    nb::object clear;
    nb::object init;

    GraphNodeImpl(nb::object type, nb::object flatten, nb::object set_key, nb::object pop_key, nb::object create_empty, nb::object clear, nb::object init)
        : NodeImplBase(type, flatten), set_key(set_key), pop_key(pop_key), create_empty(create_empty), clear(clear), init(init) {}

    bool __eq__(const nb::object &other_obj) const
    {
      if (!nb::isinstance<GraphNodeImpl>(other_obj))
      {
        return false;
      }
      GraphNodeImpl other = nb::cast<GraphNodeImpl>(other_obj);
      return type.equal(other.type) && flatten.equal(other.flatten) && set_key.equal(other.set_key) && pop_key.equal(other.pop_key) && create_empty.equal(other.create_empty) && clear.equal(other.clear) && init.equal(other.init);
    }

    int __hash__() const
    {
      return nb::hash(nb::make_tuple(type, flatten, set_key, pop_key, create_empty, clear, init));
    }

    nb::tuple __getstate__() const
    {
      return nb::make_tuple(type, flatten, set_key, pop_key, create_empty, clear, init);
    }

    static void __setstate__(GraphNodeImpl &graphnodeimpl, nb::tuple &t)
    {
      new (&graphnodeimpl) GraphNodeImpl(t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
    }
  };

  // @dataclasses.dataclass(frozen=True, slots=True)
  // class PytreeNodeImpl(NodeImplBase[Node, Leaf, AuxData]):
  //   unflatten: tp.Callable[[tp.Sequence[tuple[Key, Leaf]], AuxData], Node]

  struct PytreeNodeImpl : public NodeImplBase
  {
    nb::object unflatten;

    PytreeNodeImpl(nb::object type, nb::object flatten, nb::object unflatten)
        : NodeImplBase(type, flatten), unflatten(unflatten) {}

    bool __eq__(const nb::object &other_obj) const
    {
      if (!nb::isinstance<PytreeNodeImpl>(other_obj))
      {
        return false;
      }
      PytreeNodeImpl other = nb::cast<PytreeNodeImpl>(other_obj);
      return unflatten.equal(other.unflatten);
    }

    int __hash__() const
    {
      return nb::hash(nb::make_tuple(type, flatten, unflatten));
    }

    nb::tuple __getstate__() const
    {
      return nb::make_tuple(type, flatten, unflatten);
    }

    static void __setstate__(PytreeNodeImpl &pytreenodeimpl, nb::tuple &t)
    {
      new (&pytreenodeimpl) PytreeNodeImpl(t[0], t[1], t[2]);
    }
  };

  //---------------------------------------------------------------
  // IndexMap
  //---------------------------------------------------------------

  struct IndexMap : public std::unordered_map<int, nb::object>
  {
  };

  //---------------------------------------------------------------
  // RefMap
  //---------------------------------------------------------------

  struct RefMapKeysIterator
  {
    std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator it;
    std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator end;

    RefMapKeysIterator(std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator it, std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator end)
        : it(it), end(end) {}

    nb::object __next__()
    {
      if (it == end)
      {
        throw nb::stop_iteration();
      }
      auto elem = it->second;
      ++it;
      return std::get<0>(elem);
    }
  };

  struct RefMapItemsIterator
  {
    std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator it;
    std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator end;

    RefMapItemsIterator(std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator it, std::unordered_map<intptr_t, std::tuple<nb::object, int>>::iterator end)
        : it(it), end(end) {}

    RefMapItemsIterator __iter__()
    {
      return *this;
    }

    nb::tuple __next__()
    {
      if (it == end)
      {
        throw nb::stop_iteration();
      }
      auto elem = it->second;
      ++it;
      return nb::make_tuple(std::get<0>(elem), std::get<1>(elem));
    }
  };

  struct RefMap
  {
    std::unordered_map<intptr_t, std::tuple<nb::object, int>> mapping;

    RefMap() {}

    RefMap(const nb::object &iterable) : RefMap()
    {
      for (auto item : iterable)
      {
        nb::object obj = item[0];
        auto value = nb::cast<int>(item[1]);
        mapping[nb_id(obj)] = {obj, value};
      }
    }

    int __getitem__(const nb::object &key)
    {
      return std::get<1>(mapping[nb_id(key)]);
    }

    void __setitem__(const nb::object &key, int value)
    {
      mapping[nb_id(key)] = std::make_tuple(key, value);
    }

    int __len__() const
    {
      return mapping.size();
    }

    bool __contains__(const nb::object &key) const
    {
      return mapping.find(nb_id(key)) != mapping.end();
    }

    RefMapKeysIterator __iter__()
    {
      return RefMapKeysIterator(mapping.begin(), mapping.end());
    };

    RefMapItemsIterator items()
    {
      return RefMapItemsIterator(mapping.begin(), mapping.end());
    }

    std::optional<int> get(const nb::object &key, std::optional<int> default_value = std::nullopt)
    {
      auto it = mapping.find(nb_id(key));
      if (it != mapping.end())
      {
        return std::get<1>(it->second);
      }
      return default_value;
    }

    void update(const RefMap &other)
    {
      for (const auto &[key, value_index] : other.mapping)
      {
        mapping[key] = value_index;
      }
    }
  };

  static IndexMap indexmap_from_refmap(const RefMap &refmap)
  {
    IndexMap indexmap;
    for (const auto &[_, value_index] : refmap.mapping)
    {
      nb::object value = std::get<0>(value_index);
      int index = std::get<1>(value_index);
      indexmap[index] = value;
    }
    return indexmap;
  };

  static RefMap refmap_from_indexmap(const IndexMap &indexmap)
  {
    RefMap refmap;
    for (const auto &[index, value] : indexmap)
    {
      refmap.mapping[nb_id(value)] = std::make_tuple(value, index);
    }
    return refmap;
  };

  //---------------------------------------------------------------
  // NodeDef
  //---------------------------------------------------------------

  struct NodeDef
  {
    nb::object type;
    std::optional<int> index;
    std::optional<int> outer_index;
    int num_attributes;
    nb::object metadata;

    NodeDef(nb::object type, std::optional<int> index, std::optional<int> outer_index, int num_attributes, nb::object metadata)
        : type(type), index(index), outer_index(outer_index), num_attributes(num_attributes), metadata(metadata) {}

    NodeDef with_no_outer_index() const
    {
      return NodeDef(type, index, std::nullopt, num_attributes, metadata);
    }

    NodeDef with_same_outer_index() const
    {
      return NodeDef(type, index, index, num_attributes, metadata);
    }

    bool __eq__(const nb::object &other_obj) const
    {
      if (!nb::isinstance<NodeDef>(other_obj))
      {
        return false;
      }
      NodeDef other = nb::cast<NodeDef>(other_obj);
      return type.equal(other.type) && index == other.index && outer_index == other.outer_index && num_attributes == other.num_attributes && metadata.equal(other.metadata);
    }

    int __hash__() const
    {
      // return nb::hash(type) ^ nb::hash(nb::cast(index)) ^ nb::hash(nb::cast(outer_index)) ^ nb::hash(nb::cast(num_attributes)) ^ nb::hash(metadata);
      return nb::hash(nb::make_tuple(type, index, outer_index, num_attributes, metadata));
    }

    nb::tuple __getstate__() const
    {
      return nb::make_tuple(type, index, outer_index, num_attributes, metadata);
    }

    static void __setstate__(NodeDef &nodedef, nb::tuple &t)
    {
      new (&nodedef) NodeDef(t[0], nb::cast<std::optional<int>>(t[1]), nb::cast<std::optional<int>>(t[2]), nb::cast<int>(t[3]), t[4]);
    }
  };

  //---------------------------------------------------------------
  // VariableDef
  //---------------------------------------------------------------

  struct VariableDef
  {
    nb::object type;
    int index;
    std::optional<int> outer_index;
    nb::object metadata;

    VariableDef(nb::object type, int index, std::optional<int> outer_index, nb::object metadata)
        : type(type), index(index), outer_index(outer_index), metadata(metadata) {}

    VariableDef with_no_outer_index() const
    {
      return VariableDef(type, index, std::nullopt, metadata);
    }

    VariableDef with_same_outer_index() const
    {
      return VariableDef(type, index, index, metadata);
    }

    bool __eq__(const nb::object &other_obj) const
    {
      if (!nb::isinstance<VariableDef>(other_obj))
      {
        return false;
      }
      VariableDef other = nb::cast<VariableDef>(other_obj);
      return type.equal(other.type) && index == other.index && outer_index == other.outer_index && metadata.equal(other.metadata);
    }

    int __hash__() const
    {
      // return nb::hash(type) ^ nb::hash(nb::cast(index)) ^ nb::hash(nb::cast(outer_index)) ^ nb::hash(metadata);
      return nb::hash(nb::make_tuple(type, index, outer_index, metadata));
    }

    nb::tuple __getstate__() const
    {
      return nb::make_tuple(type, index, outer_index, metadata);
    }

    static void __setstate__(VariableDef &variabledef, nb::tuple &t)
    {
      new (&variabledef) VariableDef(t[0], nb::cast<int>(t[1]), nb::cast<std::optional<int>>(t[2]), t[3]);
    }
  };

  //---------------------------------------------------------------
  // NodeRef
  //---------------------------------------------------------------

  struct NodeRef
  {
    int index;

    NodeRef(int index)
        : index(index) {}

    bool __eq__(const nb::object &other_obj) const
    {
      if (!nb::isinstance<NodeRef>(other_obj))
      {
        return false;
      }
      NodeRef other = nb::cast<NodeRef>(other_obj);
      return index == other.index;
    }

    int __hash__() const
    {
      return nb::hash(nb::cast(index));
    }

    nb::tuple __getstate__() const
    {
      return nb::make_tuple(index);
    }

    static void __setstate__(NodeRef &noderef, nb::tuple &t)
    {
      new (&noderef) NodeRef(nb::cast<int>(t[0]));
    }
  };

  NB_MODULE(flaxlib_cpp, m)
  {
    nb::class_<NodeImplBase>(m, "NodeImplBase")
        .def(nb::init<nb::object, nb::object>())
        .def_prop_ro("type", [](const NodeImplBase &n)
                     { return n.type; })
        .def_prop_ro("flatten", [](const NodeImplBase &n)
                     { return n.flatten; })
        .def("node_dict", &NodeImplBase::node_dict, nb::arg().none())
        .def("__eq__", &NodeImplBase::__eq__, nb::arg().none())
        .def("__hash__", &NodeImplBase::__hash__)
        .def("__getstate__", &NodeImplBase::__getstate__)
        .def("__setstate__", &NodeImplBase::__setstate__);

    nb::class_<GraphNodeImpl>(m, "GraphNodeImpl")
        .def(nb::init<nb::object, nb::object, nb::object, nb::object, nb::object, nb::object, nb::object>())
        .def_prop_ro("type", [](const GraphNodeImpl &n)
                     { return n.type; })
        .def_prop_ro("flatten", [](const GraphNodeImpl &n)
                     { return n.flatten; })
        .def_prop_ro("set_key", [](const GraphNodeImpl &n)
                     { return n.set_key; })
        .def_prop_ro("pop_key", [](const GraphNodeImpl &n)
                     { return n.pop_key; })
        .def_prop_ro("create_empty", [](const GraphNodeImpl &n)
                     { return n.create_empty; })
        .def_prop_ro("clear", [](const GraphNodeImpl &n)
                     { return n.clear; })
        .def_prop_ro("init", [](const GraphNodeImpl &n)
                     { return n.init; })
        .def("node_dict", &GraphNodeImpl::node_dict, nb::arg().none())
        .def("__eq__", &GraphNodeImpl::__eq__, nb::arg().none())
        .def("__hash__", &GraphNodeImpl::__hash__)
        .def("__getstate__", &GraphNodeImpl::__getstate__)
        .def("__setstate__", &GraphNodeImpl::__setstate__);

    nb::class_<PytreeNodeImpl>(m, "PytreeNodeImpl")
        .def(nb::init<nb::object, nb::object, nb::object>())
        .def_prop_ro("type", [](const PytreeNodeImpl &n)
                     { return n.type; })
        .def_prop_ro("flatten", [](const PytreeNodeImpl &n)
                     { return n.flatten; })
        .def_prop_ro("unflatten", [](const PytreeNodeImpl &n)
                     { return n.unflatten; })
        .def("node_dict", &PytreeNodeImpl::node_dict, nb::arg().none())
        .def("__eq__", &PytreeNodeImpl::__eq__, nb::arg().none())
        .def("__hash__", &PytreeNodeImpl::__hash__)
        .def("__getstate__", &PytreeNodeImpl::__getstate__)
        .def("__setstate__", &PytreeNodeImpl::__setstate__);

    nb::bind_map<IndexMap>(m, "IndexMap")
        .def_static("from_refmap", &indexmap_from_refmap);
    nb::class_<flaxlib::RefMapKeysIterator>(m, "RefMapKeysIterator")
        .def("__next__", &flaxlib::RefMapKeysIterator::__next__);

    nb::class_<flaxlib::RefMapItemsIterator>(m, "RefMapItemsIterator")
        .def("__iter__", &flaxlib::RefMapItemsIterator::__iter__)
        .def("__next__", &flaxlib::RefMapItemsIterator::__next__);

    nb::class_<flaxlib::RefMap>(m, "RefMap")
        .def(nb::init<>())
        .def(nb::init<nb::object>())
        .def_static("from_indexmap", &refmap_from_indexmap)
        .def("__getitem__", &flaxlib::RefMap::__getitem__, nb::arg())
        .def("__setitem__", &flaxlib::RefMap::__setitem__, nb::arg(), nb::arg())
        .def("__len__", &flaxlib::RefMap::__len__)
        .def("__contains__", &flaxlib::RefMap::__contains__, nb::arg())
        .def("__iter__", &flaxlib::RefMap::__iter__)
        .def("items", &flaxlib::RefMap::items)
        .def("get", &flaxlib::RefMap::get, nb::arg(), nb::arg().none())
        .def("update", &flaxlib::RefMap::update);

    nb::class_<flaxlib::NodeDef>(m, "NodeDef")
        .def(nb::init<nb::object, std::optional<int>, std::optional<int>, int, nb::object>(),
             nb::arg(), nb::arg().none(), nb::arg().none(), nb::arg(), nb::arg().none())
        .def_prop_ro("type", [](const flaxlib::NodeDef &n)
                     { return n.type; })
        .def_prop_ro("index", [](const flaxlib::NodeDef &n)
                     { return n.index; })
        .def_prop_ro("outer_index", [](const flaxlib::NodeDef &n)
                     { return n.outer_index; })
        .def_prop_ro("num_attributes", [](const flaxlib::NodeDef &n)
                     { return n.num_attributes; })
        .def_prop_ro("metadata", [](const flaxlib::NodeDef &n)
                     { return n.metadata; })
        .def("with_no_outer_index", &flaxlib::NodeDef::with_no_outer_index)
        .def("with_same_outer_index", &flaxlib::NodeDef::with_same_outer_index)
        .def("__eq__", &flaxlib::NodeDef::__eq__, nb::arg().none())
        .def("__hash__", &flaxlib::NodeDef::__hash__)
        .def("__getstate__", &flaxlib::NodeDef::__getstate__)
        .def("__setstate__", &flaxlib::NodeDef::__setstate__);

    nb::class_<flaxlib::VariableDef>(m, "VariableDef")
        .def(nb::init<nb::object, int, std::optional<int>, nb::object>(),
             nb::arg(), nb::arg(), nb::arg().none(), nb::arg().none())
        .def_prop_ro("type", [](const flaxlib::VariableDef &n)
                     { return n.type; })
        .def_prop_ro("index", [](const flaxlib::VariableDef &n)
                     { return n.index; })
        .def_prop_ro("outer_index", [](const flaxlib::VariableDef &n)
                     { return n.outer_index; })
        .def_prop_ro("metadata", [](const flaxlib::VariableDef &n)
                     { return n.metadata; })
        .def("with_no_outer_index", &flaxlib::VariableDef::with_no_outer_index)
        .def("with_same_outer_index", &flaxlib::VariableDef::with_same_outer_index)
        .def("__eq__", &flaxlib::VariableDef::__eq__, nb::arg().none())
        .def("__hash__", &flaxlib::VariableDef::__hash__)
        .def("__getstate__", &flaxlib::VariableDef::__getstate__)
        .def("__setstate__", &flaxlib::VariableDef::__setstate__);

    nb::class_<flaxlib::NodeRef>(m, "NodeRef")
        .def(nb::init<int>(),
             nb::arg())
        .def_prop_ro("index", [](const flaxlib::NodeRef &n)
                     { return n.index; })
        .def("__eq__", &flaxlib::NodeRef::__eq__, nb::arg().none())
        .def("__hash__", &flaxlib::NodeRef::__hash__)
        .def("__getstate__", &flaxlib::NodeRef::__getstate__)
        .def("__setstate__", &flaxlib::NodeRef::__setstate__);
  }
} // namespace flaxlib