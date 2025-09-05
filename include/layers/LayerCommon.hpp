#pragma once
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <InstanceStorage.hpp>
#include <ActivationFuncs.hpp>
#include <Buffers.hpp>
#include <Linalg.hpp>
#include <ModelConfig.hpp>

//Tag for all "leaf" types that can be used to build a model tree
struct LeafTag{};

template <class...> using void_type = void;

template <class T, class = void>
struct is_leaf : std::false_type {};

//Specialize only when T::tag exists
template <class T>
struct is_leaf<T, void_type<typename std::decay_t<T>::tag> >
  : std::is_same<typename std::decay_t<T>::tag, LeafTag> {};

template <class T>
constexpr bool is_leaf_v = is_leaf<T>::value;

#define ISLEAF(a) is_leaf_v<a>
#define FLOATTYPE(a) typename std::decay<a>::type::FloatType
#define INPUTTYPE(a) typename std::decay<a>::type::InputType
#define LAYEROUTPUTTYPE(a) typename std::decay<decltype( std::declval<typename std::decay<a>::type&>().value( std::declval<INPUTTYPE(a)>() ) )>::type
#define LAYERTYPEOUTPUTTYPE(a) typename std::decay<decltype( std::declval<a>().value( std::declval<typename a::InputType>() ) )>::type
#define CONFIGTYPE(a) typename std::decay<a>::type::ModelConfig




