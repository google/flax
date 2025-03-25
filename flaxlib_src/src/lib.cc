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

    void update(const RefMap &other)
    {
      for (const auto &[key_id, value_tuple] : other.mapping)
      {
        mapping[key_id] = value_tuple;
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
    nb::bind_map<IndexMap>(m, "IndexMap")
        .def_static("from_refmap", &indexmap_from_refmap);
    nb::class_<flaxlib::RefMapKeysIterator>(m, "RefMapKeysIterator")
        .def("__next__", &flaxlib::RefMapKeysIterator::__next__);

    nb::class_<flaxlib::RefMapItemsIterator>(m, "RefMapItemsIterator")
        .def("__iter__", &flaxlib::RefMapItemsIterator::__iter__)
        .def("__next__", &flaxlib::RefMapItemsIterator::__next__);

    nb::class_<flaxlib::RefMap>(m, "RefMap")
        .def(nb::init<>())
        .def(nb::init<nb::object>(), nb::arg("iterable"))
        .def("update", &flaxlib::RefMap::update)
        .def_static("from_indexmap", &refmap_from_indexmap)
        .def("__getitem__", &flaxlib::RefMap::__getitem__, nb::arg("key").none())
        .def("__setitem__", &flaxlib::RefMap::__setitem__, nb::arg("key").none(), nb::arg("value"))
        .def("__len__", &flaxlib::RefMap::__len__)
        .def("__contains__", &flaxlib::RefMap::__contains__, nb::arg("key").none())
        .def("__iter__", &flaxlib::RefMap::__iter__)
        .def("items", &flaxlib::RefMap::items)
        .def("get", &flaxlib::RefMap::get, nb::arg("key").none(), nb::arg("default_value").none());

    nb::class_<flaxlib::NodeDef>(m, "NodeDef")
        .def(nb::init<nb::object, std::optional<int>, std::optional<int>, int, nb::object>(),
             nb::arg("type"), nb::arg("index").none(), nb::arg("outer_index").none(), nb::arg("num_attributes"), nb::arg("metadata").none())
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
             nb::arg("type"), nb::arg("index"), nb::arg("outer_index").none(), nb::arg("metadata").none())
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
             nb::arg("index"))
        .def_prop_ro("index", [](const flaxlib::NodeRef &n)
                     { return n.index; })
        .def("__eq__", &flaxlib::NodeRef::__eq__, nb::arg().none())
        .def("__hash__", &flaxlib::NodeRef::__hash__)
        .def("__getstate__", &flaxlib::NodeRef::__getstate__)
        .def("__setstate__", &flaxlib::NodeRef::__setstate__);
  }
} // namespace flaxlib