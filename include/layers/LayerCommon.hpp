#pragma once
#include <type_traits>
#include <sstream>
#include <Tensors.hpp>
#include <InstanceStorage.hpp>
#include <ActivationFuncs.hpp>
#include <RingBuffer.hpp>
#include <Linalg.hpp>

//Tag for all "leaf" types that can be used to build a model tree
struct LeafTag{};
#define ISLEAF(a) std::is_same<typename std::decay<a>::type::tag,LeafTag>::value
#define FLOATTYPE(a) typename std::decay<a>::type::FloatType
#define INPUTTYPE(a) typename std::decay<a>::type::InputType
#define LAYEROUTPUTTYPE(a) typename std::decay<decltype( std::declval<typename std::decay<a>::type&>().value( std::declval<INPUTTYPE(a)>() ) )>::type
#define LAYERTYPEOUTPUTTYPE(a) typename std::decay<decltype( std::declval<a>().value( std::declval<typename a::InputType>() ) )>::type
