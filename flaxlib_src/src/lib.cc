#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <map>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <nanobind/make_iterator.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace flaxlib
{
  // -----------------------------------
  // helper functions
  // -----------------------------------
  intptr_t get_id(nb::object obj)
  {
    // Get the object ID
    return reinterpret_cast<intptr_t>(obj.ptr());
  }

  bool nb_isinstance(nanobind::handle inst, nanobind::handle cls)
  {
    int ret = PyObject_IsInstance(inst.ptr(), cls.ptr());
    if (ret == -1)
    {
      throw nb::python_error();
    }
    return ret;
  }

  nb::object vector_to_tuple(const std::vector<nb::object> &vec)
  {

    if (vec.empty())
    {
      return nb::tuple();
    }
    else
    {
      auto ls = nb::list();
      for (const auto &item : vec)
      {
        ls.append(item);
      }
      auto result = nb::tuple(ls);
      return result;
    }
  }

  // -----------------------------------
  // IndexMapping
  // -----------------------------------
  class IndexMappingKeysIterator
  {
  public:
    IndexMappingKeysIterator(const std::unordered_map<int, int> &data) : it(data.begin()), end(data.end()) {}

    int next()
    {
      if (it == end)
      {
        throw nb::stop_iteration();
      }

      return it++->first;
    }

    IndexMappingKeysIterator &__iter__()
    {
      return *this;
    }

  private:
    std::unordered_map<int, int>::const_iterator it;
    std::unordered_map<int, int>::const_iterator end;
  };

  struct IndexMapping
  {
    std::unordered_map<int, int> mapping;

    IndexMapping(std::unordered_map<int, int> &mapping, bool copy)
    {
      if (copy)
      {
        this->mapping = mapping;
      }
      else
      {
        this->mapping = std::move(mapping);
      }
    }

    // define the python __hash__ method
    uint64_t __hash__()
    {
      EVP_MD_CTX *mdctx;
      const EVP_MD *md;
      unsigned char md_value[EVP_MAX_MD_SIZE];
      unsigned int md_len;

      // Serialize the map
      std::stringstream ss;
      for (const auto &pair : mapping)
      {
        ss << pair.first << ":" << pair.second << ",";
      }
      std::string serializedData = ss.str();

      OpenSSL_add_all_digests();

      md = EVP_get_digestbyname("SHA256");
      if (!md)
      {
        throw std::runtime_error("Unknown message digest BLAKE3");
      }

      mdctx = EVP_MD_CTX_new();
      EVP_DigestInit_ex(mdctx, md, NULL);
      EVP_DigestUpdate(mdctx, serializedData.c_str(), serializedData.size());
      EVP_DigestFinal_ex(mdctx, md_value, &md_len);
      EVP_MD_CTX_free(mdctx);

      // Convert (part of) the digest to a 64-bit integer
      uint64_t result = 0;
      for (size_t i = 0; i < 8 && i < md_len; ++i)
      {
        result = (result << 8) | static_cast<uint8_t>(md_value[i]);
      }

      return result;
    }

    // define the python __repr__ method
    std::string __repr__()
    {
      std::string repr;
      if (mapping.size() == 1)
      {
        repr = "IndexMapping({";
        for (const auto &pair : mapping)
        {
          repr += std::to_string(pair.first) + ": " + std::to_string(pair.second);
        }
        repr += "})";
      }
      else
      {
        repr = "IndexMapping({\n";
        for (const auto &pair : mapping)
        {
          repr += "  " + std::to_string(pair.first) + ": " + std::to_string(pair.second) + ",\n";
        }
        if (!mapping.empty())
        {
          repr.pop_back();
          repr.pop_back();
        }
        repr += "\n})";
      }
      return repr;
    }

    // define the python __getitem__ method
    int __getitem__(int key) const
    {
      return mapping.at(key);
    }

    // define __iter__ method
    IndexMappingKeysIterator __iter__() const
    {
      return IndexMappingKeysIterator(mapping);
    }

    // define the python __len__ method
    size_t __len__() const
    {
      return mapping.size();
    }

    // define the python __contains__ method
    bool __contains__(int key) const
    {
      return mapping.find(key) != mapping.end();
    }

    bool __eq__(const nb::object &other) const
    {
      if (!nb::isinstance<IndexMapping>(other))
      {
        return false;
      }

      auto other_mapping = nb::cast<IndexMapping>(other);
      return mapping == other_mapping.mapping;
    }

    nb::object items() const
    {
      return nb::make_iterator(
          nb::type<std::vector<std::pair<int, int>>>(), "IndexMappingItemsIterator", mapping.begin(), mapping.end());
    }
  };

  // -----------------------------------
  // RefIndexMapping
  // -----------------------------------

  struct RefIndexMappingKeysIterator
  {
  public:
    RefIndexMappingKeysIterator(const std::unordered_map<intptr_t, std::pair<nb::object, int>> &data) : it(data.begin()), end(data.end()) {}

    nb::object next()
    {
      if (it == end)
      {
        throw nb::stop_iteration();
      }

      return it++->second.first;
    }

    RefIndexMappingKeysIterator &__iter__()
    {
      return *this;
    }

  private:
    std::unordered_map<intptr_t, std::pair<nb::object, int>>::const_iterator it;
    std::unordered_map<intptr_t, std::pair<nb::object, int>>::const_iterator end;
  };

  struct RefIndexMappingItemsIterator
  {
  public:
    RefIndexMappingItemsIterator(const std::unordered_map<intptr_t, std::pair<nb::object, int>> &data) : it(data.begin()), end(data.end()) {}

    std::pair<nb::object, int> next()
    {
      if (it == end)
      {
        throw nb::stop_iteration();
      }

      return it++->second;
    }

    RefIndexMappingItemsIterator &__iter__()
    {
      return *this;
    }

  private:
    std::unordered_map<intptr_t, std::pair<nb::object, int>>::const_iterator it;
    std::unordered_map<intptr_t, std::pair<nb::object, int>>::const_iterator end;
  };

  struct RefIndexMapping
  {
    std::unordered_map<intptr_t, std::pair<nb::object, int>> mapping;

    RefIndexMapping(std::map<nb::object, int> ref_mapping)
    {
      for (const auto &pair : ref_mapping)
      {
        mapping[get_id(pair.first)] = {pair.first, pair.second};
      }
    }

    int __getitem__(nb::object key) const
    {
      return mapping.at(get_id(key)).second;
    }

    bool __contains__(nb::object key) const
    {
      return mapping.find(get_id(key)) != mapping.end();
    }

    void __setitem__(nb::object key, int value)
    {
      mapping[get_id(key)] = {key, value};
    }

    void __delitem__(nb::object key)
    {
      mapping.erase(get_id(key));
    }

    RefIndexMappingKeysIterator __iter__() const
    {
      return RefIndexMappingKeysIterator(mapping);
    }

    size_t __len__() const
    {
      return mapping.size();
    }

    // __repr__ method
    std::string __repr__()
    {
      std::string repr;
      if (mapping.size() == 1)
      {
        repr = "RefIndexMapping({";
        for (const auto &pair : mapping)
        {
          repr += nb::cast<std::string>(nb::repr(pair.second.first)) + ": " + std::to_string(pair.second.second);
        }
        repr += "})";
      }
      else
      {
        repr = "RefIndexMapping({\n";
        for (const auto &pair : mapping)
        {
          repr += "  " + nb::cast<std::string>(nb::repr(pair.second.first)) + ": " + std::to_string(pair.second.second) + ",\n";
        }
        if (!mapping.empty())
        {
          repr.pop_back();
          repr.pop_back();
        }
        repr += "\n})";
      }
      return repr;
    }

    RefIndexMappingItemsIterator items() const
    {
      return RefIndexMappingItemsIterator(mapping);
    }
  };

  // -------------------------------------
  // IndexRefMapping
  // -------------------------------------

  struct IndexRefMappingKeysIterator
  {
  public:
    IndexRefMappingKeysIterator(const std::unordered_map<int, nb::object> &data) : it(data.begin()), end(data.end()) {}

    int next()
    {
      if (it == end)
      {
        throw nb::stop_iteration();
      }

      return get_id(it++->second);
    }

    IndexRefMappingKeysIterator &__iter__()
    {
      return *this;
    }

  private:
    std::unordered_map<int, nb::object>::const_iterator it;
    std::unordered_map<int, nb::object>::const_iterator end;
  };

  struct IndexRefMapping
  {
    std::unordered_map<int, nb::object> mapping;

    IndexRefMapping(std::unordered_map<int, nb::object> mapping) : mapping(mapping) {}

    nb::object __getitem__(int key) const
    {
      return mapping.at(key);
    }

    bool __contains__(int key) const
    {
      return mapping.find(key) != mapping.end();
    }

    void __setitem__(int key, nb::object value)
    {
      mapping[key] = value;
    }

    void __delitem__(int key)
    {
      mapping.erase(key);
    }

    IndexRefMappingKeysIterator __iter__() const
    {
      return IndexRefMappingKeysIterator(mapping);
    }

    size_t __len__() const
    {
      return mapping.size();
    }

    std::string __repr__()
    {
      std::string repr;
      if (mapping.size() <= 1)
      {
        repr = "IndexRefMapping({";
        for (const auto &pair : mapping)
        {
          repr += std::to_string(pair.first) + ": " + nb::cast<std::string>(nb::repr(pair.second));
        }
        repr += "})";
      }
      else
      {
        repr = "IndexRefMapping({\n";
        for (const auto &pair : mapping)
        {
          repr += "  " + std::to_string(pair.first) + ": " + nb::cast<std::string>(nb::repr(pair.second)) + ",\n";
        }
        if (!mapping.empty())
        {
          repr.pop_back();
          repr.pop_back();
        }
        repr += "\n})";
      }
      return repr;
    }

    nb::object items() const
    {
      return nb::make_iterator(nb::type<std::vector<std::pair<int, nb::object>>>(), "IndexRefMappingItemsIterator", mapping.begin(), mapping.end());
    }
  };

  // -------------------------------------
  // functions
  // -------------------------------------

  IndexRefMapping create_index_ref(RefIndexMapping ref_index, IndexMapping index_mapping)
  {
    std::unordered_map<int, nb::object> new_mapping;
    for (const auto &pair : ref_index.mapping)
    {
      auto a = pair.second.first;
      auto b = pair.second.second;

      auto b_pos = index_mapping.mapping.find(b);
      if (b_pos != index_mapping.mapping.end())
      {
        new_mapping[b_pos->second] = a;
      }
    }
    return IndexRefMapping(new_mapping);
  }

  nb::object _graph_flatten(
      std::vector<nb::object> &path,
      RefIndexMapping &ref_index,
      std::vector<std::pair<nb::object, nb::object>> &flat_state,
      nb::object node)
  {
    // import graph Module from flax.nnx
    auto graph = nb::module_::import_("flax.nnx.graph");
    auto jax = nb::module_::import_("jax");
    auto np = nb::module_::import_("numpy");

    auto jax_Array = nb::getattr(jax, "Array");
    auto np_ndarray = nb::getattr(np, "ndarray");
    auto GraphNodeImpl = nb::getattr(graph, "GraphNodeImpl");
    auto Variable = nb::getattr(graph, "Variable");
    auto SubGraphAttribute = nb::getattr(graph, "SubGraphAttribute");
    auto StaticAttribute = nb::getattr(graph, "StaticAttribute");
    auto LeafAttribute = nb::getattr(graph, "LeafAttribute");
    auto NodeRef = nb::getattr(graph, "NodeRef");
    auto NodeDef = nb::getattr(graph, "NodeDef");
    auto VariableDef = nb::getattr(graph, "VariableDef");
    auto HashableMapping = nb::getattr(graph, "HashableMapping");

    if (!nb::bool_(nb::getattr(graph, "is_node")(node)))
    {
      throw std::runtime_error("Unsupported type: " + nb::cast<std::string>(node.type().attr("__name__")) + ", this is a bug.");
    }

    if (ref_index.__contains__(node))
    {
      return NodeRef(node.type(), ref_index.__getitem__(node));
    }

    auto node_impl = nb::getattr(graph, "get_node_impl")(node);

    int index;
    // only cache graph nodes
    if (nb_isinstance(node_impl, GraphNodeImpl))
    {
      index = ref_index.__len__();
      ref_index.__setitem__(node, index);
    }
    else
    {
      index = -1;
    }

    std::vector<nb::object> attributes;

    auto values_metadata = nb::getattr(node_impl, "flatten")(node);
    auto values = values_metadata[0];
    auto metadata = values_metadata[1];

    for (const auto &key_value : values)
    {
      auto key = key_value[0];
      auto value = key_value[1];

      path.push_back(key);

      if (nb::bool_(nb::getattr(graph, "is_node")(value)))
      {
        auto nodedef = _graph_flatten(path, ref_index, flat_state, value);
        attributes.push_back(SubGraphAttribute(key, nodedef));
      }
      else if (nb_isinstance(value, Variable))
      {
        if (ref_index.__contains__(value))
        {
          attributes.push_back(LeafAttribute(key, NodeRef(value.type(), ref_index.__getitem__(value))));
        }
        else
        {
          auto path_tuple = vector_to_tuple(path);
          flat_state.push_back({path_tuple, nb::getattr(value, "to_state")()});
          auto variable_index = ref_index.__len__();
          ref_index.__setitem__(value, variable_index);
          auto var_meta = HashableMapping(nb::getattr(value, "_var_metadata"));
          auto variabledef = VariableDef(value.type(), variable_index, var_meta);
          attributes.push_back(LeafAttribute(key, variabledef));
        }
      }
      else
      {
        if (nb_isinstance(value, jax_Array) || nb_isinstance(value, np_ndarray))
        {
          std::string path_str;
          for (const auto &part : path)
          {
            path_str += nb::cast<std::string>(nb::repr(part)) + "/";
          }
          throw std::runtime_error("Arrays leaves are not supported, at " + path_str + ": " + nb::cast<std::string>(nb::repr(value)));
        }
        attributes.push_back(StaticAttribute(key, value));
      }
      path.pop_back();
    }

    auto attributes_tuple = vector_to_tuple(attributes);
    auto nodedef = nb::getattr(NodeDef, "create")(
        nb::getattr(node_impl, "type"), index, attributes_tuple, metadata, nb::none());

    return nodedef;
  }

  std::pair<nb::object, nb::list> _graph_flatten_top(
      RefIndexMapping &ref_index,
      nb::object node)
  {
    // print "here"
    std::vector<nb::object> path = {};
    std::vector<std::pair<nb::object, nb::object>> flat_state = {};
    auto nodedef = _graph_flatten(path, ref_index, flat_state, node);

    auto flat_state_out = nb::list();
    for (const auto &pair : flat_state)
    {
      flat_state_out.append(nb::make_tuple(pair.first, pair.second));
    }
    return {nodedef, flat_state_out};
  }

  NB_MODULE(flaxlib_cpp, m)
  {
    //-------------------------------------------------------------------------
    // IndexMapping
    //-------------------------------------------------------------------------
    nb::class_<IndexMapping>(m, "IndexMapping")
        .def(nb::init<std::unordered_map<int, int> &, bool>(), nb::arg("mapping"), nb::arg("copy") = true)
        .def("__hash__", &IndexMapping::__hash__)
        .def("__repr__", &IndexMapping::__repr__)
        .def("__getitem__", &IndexMapping::__getitem__)
        .def("__iter__", &IndexMapping::__iter__)
        .def("__len__", &IndexMapping::__len__)
        .def("__contains__", &IndexMapping::__contains__, nb::arg("key").none())
        .def("__eq__", &IndexMapping::__eq__)
        .def("items", &IndexMapping::items);

    nb::class_<IndexMappingKeysIterator>(m, "IndexMappingIterator")
        .def("__iter__", &IndexMappingKeysIterator::__iter__)
        .def("__next__", &IndexMappingKeysIterator::next);

    //-------------------------------------------------------------------------
    // RefIndexMapping
    //-------------------------------------------------------------------------
    nb::class_<RefIndexMapping>(m, "RefIndexMapping")
        .def(nb::init<std::map<nb::object, int>>())
        .def("__getitem__", &RefIndexMapping::__getitem__)
        .def("__contains__", &RefIndexMapping::__contains__, nb::arg("key").none())
        .def("__setitem__", &RefIndexMapping::__setitem__)
        .def("__delitem__", &RefIndexMapping::__delitem__)
        .def("__iter__", &RefIndexMapping::__iter__)
        .def("__len__", &RefIndexMapping::__len__)
        .def("__repr__", &RefIndexMapping::__repr__)
        .def("items", &RefIndexMapping::items);

    nb::class_<RefIndexMappingKeysIterator>(m, "RefIndexMappingKeysIterator")
        .def("__iter__", &RefIndexMappingKeysIterator::__iter__)
        .def("__next__", &RefIndexMappingKeysIterator::next);

    nb::class_<RefIndexMappingItemsIterator>(m, "RefIndexMappingItemsIterator")
        .def("__iter__", &RefIndexMappingItemsIterator::__iter__)
        .def("__next__", &RefIndexMappingItemsIterator::next);

    //-------------------------------------------------------------------------
    // IndexRefMapping
    //-------------------------------------------------------------------------
    nb::class_<IndexRefMapping>(m, "IndexRefMapping")
        .def(nb::init<const std::unordered_map<int, nb::object> &>())
        .def("__getitem__", &IndexRefMapping::__getitem__)
        .def("__contains__", &IndexRefMapping::__contains__, nb::arg("key").none())
        .def("__setitem__", &IndexRefMapping::__setitem__)
        .def("__delitem__", &IndexRefMapping::__delitem__)
        .def("__iter__", &IndexRefMapping::__iter__)
        .def("__len__", &IndexRefMapping::__len__)
        .def("__repr__", &IndexRefMapping::__repr__)
        .def("items", &IndexRefMapping::items);

    nb::class_<IndexRefMappingKeysIterator>(m, "IndexRefMappingKeysIterator")
        .def("__iter__", &IndexRefMappingKeysIterator::__iter__)
        .def("__next__", &IndexRefMappingKeysIterator::next);

    //-------------------------------------------------------------------------
    // functions
    //-------------------------------------------------------------------------
    m.def("create_index_ref", &create_index_ref);
    m.def("_graph_flatten_top", &_graph_flatten_top);
    m.def("_graph_flatten", &_graph_flatten);
  }

} // namespace flaxlib