// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//! \file MLOperatorAuthor.h
#pragma once

#if defined(__cplusplus)
#if (!defined(_MSC_VER)) || (_MSC_VER >= 1700)

#if !defined(COM_NO_WINDOWS_H)
#include <unknwn.h>
#endif /* !defined(COM_NO_WINDOWS_H) */

#include <cstdint>
#include <winapifamily.h>

#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP)

static_assert(sizeof(bool) == 1, "Unsupported size for bool type");


//! \enum MLOperatorAttributeType
//! \brief Specifies the type of an attribute.
enum class MLOperatorAttributeType : uint32_t 
{
    //! Undefined (unused)
    Undefined = 0,

    //! 32 bit floating point
    Float = 2,

    //! 64 bit integer
    Int = 3,

    //! String
    String = 4,

    //! Array of 32 bit floating point values
    FloatArray = 7,

    //! Array of 64 bit floating integer values
    IntArray = 8,

    //! Array of string values
    StringArray = 9
};

//! \enum MLOperatorTensorDataType
//! \brief Specifies the data type of a tensor.
//! Each data type numerically matches corresponding ONNX types.
enum class MLOperatorTensorDataType : uint32_t
{
    //! Undefined (unused).
    Undefined = 0,

    //! IEEE 32 bit floating point 
    Float = 1,

    //! 8 bit unsigned integer
    UInt8 = 2,

    //! 8 bit signed integer
    Int8 = 3,

    //! 16 bit unsigned integer
    UInt16 = 4,

    //! 16 bit signed integer
    Int16 = 5,

    //! 32 bit signed integer
    Int32 = 6,

    //! 64 bit signed integer
    Int64 = 7,

    //! String (unsupported)
    String = 8,

    //! 8 bit boolean. Values other than zero and one result in undefined behavior.
    Bool = 9,

    //! IEEE 16 bit floating point
    Float16 = 10,

    //! 64 bit double-precision floating point
    Double = 11,

    //! 32 bit unsigned integer
    UInt32 = 12,

    //! 64 bit unsigned integer
    UInt64 = 13,

    //! 64 bit Complex type (unsupported)
    Complex64 = 14,

    //! 128 bit complex type (unsupported)
    Complex128 = 15
};

//! \enum MLOperatorEdgeType
//! \brief Specifies the types of an input or output edge of an operator.
enum class MLOperatorEdgeType : uint32_t 
{	
    Undefined = 0,
    Tensor = 1,
};
 
//! \struct MLOperatorEdgeDescription
//! \brief Specifies the properties of an input or output edge of an operator.
struct MLOperatorEdgeDescription 
{
    //! The type of the edge.
    MLOperatorEdgeType edgeType;
    
    union 
    {
        uint64_t reserved;

        //! The data type of a tensor.  Used when edgeType is set to Tensor.
        MLOperatorTensorDataType tensorDataType;
    };
};
 
//! \interface IMLOperatorAttributes
//! \brief Represents the values of an operator's attributes, as determined by a model using the operator.
//! This interface is called by implementations of custom operator kernels, and by implementations
//! of shape and type inferrers.
interface DECLSPEC_UUID("4B1B1759-EC40-466C-AAB4-BEB5347FD24C") DECLSPEC_NOVTABLE
IMLOperatorAttributes : IUnknown
{
    //! Gets the count of elements in an attribute.
    //! This may be used to determine if an attribute exists, and to determine the
    //! count of elements within an attribute of an array type.
    STDMETHOD(GetAttributeElementCount)( 
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        _Out_ uint32_t* elementCount
        ) const noexcept PURE;

    //! Gets the value of an attribute element which is of a numeric type.  
    //! For attributes which are of array types, this method queries
    //! an individual element within the attribute at the specified index.
    STDMETHOD(GetAttribute)(
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        uint32_t elementCount,
        size_t elementByteSize,
        _Out_writes_bytes_(elementCount * elementByteSize) void* value
        ) const noexcept PURE;
 
    //! Gets the length of an attribute element which is of a string type.
    //! For attributes which are string arrays, this method queries
    //! the size of an individual element within the attribute at the 
    //! specified index.
    //! The string is in UTF-8 format.  The size includes the null termination character.
    STDMETHOD(GetStringAttributeElementLength)(
        _In_z_ const char* name,
        uint32_t elementIndex,
        _Out_ uint32_t* attributeElementByteSize
        ) const noexcept PURE;
 
    //! Gets the value of an attribute element which is of a string type.
    //! For attributes which are string arrays, this method queries
    //! the value of an individual element within the attribute at the 
    //! specified index.
    //! The string is in UTF-8 format.  The size includes the null termination character.
    STDMETHOD(GetStringAttributeElement)(
        _In_z_ const char* name,
        uint32_t elementIndex,
        uint32_t attributeElementByteSize,
        _Out_writes_(attributeElementByteSize) char* attributeElement
        ) const noexcept PURE;
};

//! \interface IMLOperatorTensorShapeDescription
//! \brief Represents the set of input and output tensor shapes of an operator.
//! This interface is called by the factory objects registered to create kernels.
//! It is available to these factory objects unless corresponding kernels are 
//! registered using the MLOperatorKernelOptions::AllowDynamicInputShapes flag.
interface DECLSPEC_UUID("F20E8CBE-3B28-4248-BE95-F96FBC6E4643") DECLSPEC_NOVTABLE
IMLOperatorTensorShapeDescription : IUnknown
{
    //! Gets the number of dimensions of a tensor input of the operator.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetInputTensorDimensionCount)(
        uint32_t inputIndex, 
        _Out_ uint32_t* dimensionCount
        ) const noexcept PURE;

    //! Gets the sizes of dimensions of an input tensor of the operator.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetInputTensorShape)(
        uint32_t inputIndex, 
        uint32_t dimensionCount, 
        _Out_writes_(dimensionCount) uint32_t* dimensions
        ) const noexcept PURE;
 
    //! Returns true if output shapes may be queried using GetOutputTensorDimensionCount 
    //! and GetOutputTensorShape. This is true if the kernel was registered with a 
    //! shape inferrer.
    STDMETHOD_(bool, HasOutputShapeDescription)() const noexcept PURE;

    //! Gets the number of dimensions of a tensor output of the operator.
    //! Returns an error if the output at the specified index is not a tensor.
    STDMETHOD(GetOutputTensorDimensionCount)(
        uint32_t outputIndex, 
        _Out_ uint32_t* dimensionCount
        ) const noexcept PURE;

    //! Gets the sizes of dimensions of a tensor output of the operator.
    //! Returns an error if the output at the specified index is not a tensor.
    STDMETHOD(GetOutputTensorShape)(
        uint32_t outputIndex, 
        uint32_t dimensionCount, 
        _Out_writes_(dimensionCount) uint32_t* dimensions
        ) const noexcept PURE;
};
 
//! \interface IMLOperatorKernelCreationContext
//! \brief Provides information about an operator's usage while kernels are being created.
interface DECLSPEC_UUID("5459B53D-A0FC-4665-ADDD-70171EF7E631") DECLSPEC_NOVTABLE
IMLOperatorKernelCreationContext : public IMLOperatorAttributes 
{
    //! Gets the number of inputs to the operator.
    STDMETHOD_(uint32_t, GetInputCount)() const noexcept PURE;

    //! Gets the number of outputs to the operator.
    STDMETHOD_(uint32_t, GetOutputCount)() const noexcept PURE;
 
    //! Returns true if an input to the operator is valid.
    //! This always returns true if within GetInputCount except for optional inputs.
    STDMETHOD_(bool, IsInputValid)(uint32_t inputIndex) const noexcept PURE;

    //! Returns true if an output to the operator is valid.
    //! This always returns true if within GetOutputCount except for optional outputs.
    STDMETHOD_(bool, IsOutputValid)(uint32_t outputIndex) const noexcept PURE;

    //! Gets the description of the specified input edge of the operator.
    STDMETHOD(GetInputEdgeDescription)(
        uint32_t inputIndex, 
        _Out_ MLOperatorEdgeDescription* edgeDescription
        ) const noexcept PURE;

    //! Gets the description of the specified output edge of the operator.
    STDMETHOD(GetOutputEdgeDescription)(
        uint32_t outputIndex, 
        _Out_ MLOperatorEdgeDescription* edgeDescription
        ) const noexcept PURE;
 
    //! Returns true if the description of input and output shapes connected to
    //! operator edges may be queried using GetTensorShapeDescription.
    //! This returns true unless the operator was registered using
    //! the MLOperatorKernelOptions::AllowDynamicInputShapes flag.
    STDMETHOD_(bool, HasTensorShapeDescription)() const noexcept PURE;
 
    //! Gets the description of input and output shapes connected to
    //! operator edges.
    STDMETHOD(GetTensorShapeDescription)(
        _COM_Outptr_ IMLOperatorTensorShapeDescription** shapeDescription
        ) const noexcept PURE;
 
    //! Returns an object whose supported interfaces vary based on the kernel type.
    //! For kernels registered with MLOperatorExecutionType::Cpu, executionObject will
    //! be set to nullptr. 
    //! For kernels registered with MLOperatorExecutionType::D3D12, executionObject will
    //! support the ID3D12GraphicsCommandList interface.
    STDMETHOD_(void, GetExecutionInterface)(
        _Outptr_result_maybenull_ IUnknown** executionObject
        ) const noexcept PURE;
};
 
//! \interface IMLOperatorTensor
//! \brief Representation of a tensor used during computation of custom operator kernels.
interface DECLSPEC_UUID("7FE41F41-F430-440E-AECE-54416DC8B9DB") DECLSPEC_NOVTABLE
IMLOperatorTensor : IUnknown
{
    //! Gets the number of dimensions in the tensor.  This may be zero.
    STDMETHOD_(uint32_t, GetDimensionCount)() const noexcept PURE;

    //! Gets the size of dimensions in the tensor.
    STDMETHOD(GetShape)(
        uint32_t dimensionCount,
        _Out_writes_(dimensionCount) uint32_t* dimensions
        ) const noexcept PURE;

    //! Gets the data type of the tensor.
    STDMETHOD_(MLOperatorTensorDataType, GetTensorDataType)() const noexcept PURE;
 
    //! Indicates whether the memory used by the tensor is CPU-addressable.
    //! This is true when kernels are registered using MLOperatorExecutionType::Cpu.
    STDMETHOD_(bool, IsCpuData)() const noexcept PURE;
 
    //! Whether the contents of the tensor are represented by an interface type, 
    //! or byte-addressable memory.  This returns true when kernels are registered 
    //! using MLOperatorExecutionType::D3D12.
    STDMETHOD_(bool, IsDataInterface)() const noexcept PURE;
 
    //! Returns a pointer to byte-addressable memory for the tensor.  This may be
    //! used when IsDataInterface returns false, because the kernel was 
    //! registered using MLOperatorExecutionType::Cpu.  The data size is derived 
    //! from the tensor's shape.  It is fully packed in memory.
    STDMETHOD_(void*, GetData)() noexcept PURE; 
    
    //! Gets an interface pointer for the tensor.  This may be
    //! used when IsDataInterface returns true, because the kernel was 
    //! registered using MLOperatorExecutionType::D3D12.  The dataInterface
    //! object supports the ID3D12Resource interface, and is a GPU buffer.
    STDMETHOD_(void, GetDataInterface)(
        _Outptr_result_maybenull_ IUnknown** dataInterface
        ) noexcept PURE;
};

//! \interface IMLOperatorKernelContext
//! \brief Provides information about an operator's usage while kernels are being computed.
interface DECLSPEC_UUID("82536A28-F022-4769-9D3F-8B278F84C0C3") DECLSPEC_NOVTABLE
IMLOperatorKernelContext : IUnknown
{
    //! Gets the input tensor of the operator at the specified index.
    //! This sets tensor to nullptr for optional inputs which do not exist. 
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetInputTensor)(
        uint32_t inputIndex, 
        _COM_Outptr_result_maybenull_ IMLOperatorTensor** tensor
        ) const noexcept PURE;
 
    //! Gets the output tensor of the operator at the specified index.
    //! This sets tensor to nullptr for optional outputs which do not exist. 
    //! If the operator kernel was registered without a shape inference method, 
    //! then the overload of GetOutputTensor which consumes the tensor's shape must 
    //! be called instead. Returns an error if the output at the specified index is 
    //! not a tensor.
    STDMETHOD(GetOutputTensor)(
        uint32_t outputIndex, 
        _COM_Outptr_result_maybenull_ IMLOperatorTensor** tensor
        ) noexcept PURE;

    //! Gets the output tensor of the operator at the specified index, while declaring
    //! its shape. 
    //! This returns nullptr for optional outputs which do not exist. 
    //! If the operator kernel was registered with a shape inference method, 
    //! then the overload of GetOutputTensor which doesn't consume a shape may also
    //! be called. Returns an error if the output at the specified index is 
    //! not a tensor.
    STDMETHOD(GetOutputTensor)(
        uint32_t outputIndex,
        uint32_t dimensionCount,
        _In_reads_(dimensionCount) const uint32_t* dimensionSizes,
        _COM_Outptr_result_maybenull_ IMLOperatorTensor** tensor
        ) noexcept PURE;
    
    //! Allocates temporary data which will be usable as intermediate memory for the duration
    //! of a call to IMLOperatorKernel::Compute.  This may be used by kernels
    //! registered using MLOperatorExecutionType::D3D12.  The data
    //! object supports the ID3D12Resource interface, and is a GPU buffer.
    STDMETHOD(AllocateTemporaryData)(size_t size, _COM_Outptr_ IUnknown** data) const = 0;
 
    //! Returns an object whose supported interfaces vary based on the kernel type.
    //! For kernels registered with MLOperatorExecutionType::Cpu, executionObject will
    //! be set to nullptr. 
    //! For kernels registered with MLOperatorExecutionType::D3D12, executionObject will
    //! support the ID3D12GraphicsCommandList interface. This may be a different object
    //! than was provided to IMLOperatorKernelCreationContext::GetExecutionInterface 
    //! when the kernel instance was created.
    STDMETHOD_(void, GetExecutionInterface)(
        _Outptr_result_maybenull_ IUnknown** executionObject
        ) const noexcept PURE;
};

//! \interface IMLOperatorKernel
//! \brief Implemented by custom operator kernels.
//! A factory which creates interfaces of this interface is supplied when 
//! registering custom operator kernels using IMLOperatorKernelFactory::RegisterOperatorKernel.
interface DECLSPEC_UUID("11C4B4A0-B467-4EAA-A1A6-B961D8D0ED79") DECLSPEC_NOVTABLE
IMLOperatorKernel : IUnknown
{
    //! Computes the outputs of the kernel.  The implementation of this method
    //! should be thread-safe.  The same instance of the kernel may be computed
    //! simultaneously on different threads.
    STDMETHOD(Compute)(IMLOperatorKernelContext* context) noexcept PURE;
};
 
//! \enum MLOperatorParameterOptions
//! \brief Specifies option flags of input and output edges of operators.
//! These options are used while defining custom operator schema.
enum class MLOperatorParameterOptions : uint32_t 
{
    //! There is a single instance of the input or output.
    Single = 0,

    //! The input or output may be omitted.
    Optional = 1,

    //! The number of instances of the operator is variable.  Variadic parameters
    //! must be last among the set of inputs or outputs.
    Variadic = 2,
};

DEFINE_ENUM_FLAG_OPERATORS(MLOperatorParameterOptions);

//! \enum MLOperatorSchemaEdgeTypeFormat
//! \brief Specifies the manner in which types of input and output edges are described.
//! This is used within MLOperatorSchemaEdgeDescription while defining custom operator schema.
enum class MLOperatorSchemaEdgeTypeFormat 
{
    //! The type is defined using MLOperatorEdgeDescription.
    EdgeDescription = 0,
 
    //! The type is defined by a type string constructed as in ONNX operator schema.
    Label = 1,
};

//! \struct MLOperatorSchemaEdgeDescription
//! \brief Specifies information about an input or output edge of an operator.
//! This is used while defining custom operator schema.
struct MLOperatorSchemaEdgeDescription
{
    //! Options of the parameter, including whether it is optional or variadic.
    MLOperatorParameterOptions options;
 
    //! The manner in which the type constraints and type mapping are defined.
    MLOperatorSchemaEdgeTypeFormat typeFormat;
    union 
    {
        const void* reserved;

        //! A type label string constructed as in ONNX operator schema. For example, "T".
        //! This is used when typeFormat is MLOperatorSchemaEdgeTypeFormat::Label.
        _Field_z_ const char* typeLabel;

        //! A structure describing type support.  
        //! This is used when typeFormat is MLOperatorSchemaEdgeTypeFormat::EdgeDescription.
        MLOperatorEdgeDescription edgeDescription;
    };
};
 
//! \struct MLOperatorEdgeTypeConstraint
//! \brief Specifies constraints upon the types of edges supported in custom operator kernels 
//! and schema. The provided type label string corresponds to type labels in the ONNX 
//! specification for the same operator. For custom schema, it corresponds to type labels 
//! specified within MLOperatorSchemaEdgeDescription when registering the operator's schema.
struct MLOperatorEdgeTypeConstraint 
{
    //! The label of the type for which the constraint is being defined.
    //! This is constructed as in ONNX operator schema. For example, "T".
    _Field_z_ const char* typeLabel;
 
    //! The set of allowed types for the constraint.
    _Field_size_opt_(allowedTypeCount) const MLOperatorEdgeDescription* allowedTypes;
    uint32_t allowedTypeCount;
};

// Legacy alias.
using MLOperatorEdgeTypeConstrant = MLOperatorEdgeTypeConstraint;

//! \interface IMLOperatorShapeInferenceContext
//! \brief Provides information about an operator's usage while shape inferrers are being invoked.
interface DECLSPEC_UUID("105B6B29-5408-4A68-9959-09B5955A3492") DECLSPEC_NOVTABLE
IMLOperatorShapeInferenceContext : public IMLOperatorAttributes 
{
    //! Gets the number of inputs to the operator.
    STDMETHOD_(uint32_t, GetInputCount)() const noexcept PURE;

    //! Gets the number of outputs to the operator.
    STDMETHOD_(uint32_t, GetOutputCount)() const noexcept PURE;

    //! Returns true if an input to the operator is valid.  
    //! This always returns true except for optional inputs and invalid indices.
    STDMETHOD_(bool, IsInputValid)(uint32_t inputIndex) const noexcept PURE;

    //! Returns true if an output to the operator is valid.  
    //! This always returns true except for optional outputs and invalid indices.
    STDMETHOD_(bool, IsOutputValid)(uint32_t outputIndex) const noexcept PURE;

    //! Gets the description of the specified input edge of the operator.
    STDMETHOD(GetInputEdgeDescription)(
        uint32_t inputIndex,
        _Out_ MLOperatorEdgeDescription* edgeDescription
        ) const noexcept PURE;

    //! Gets the number of dimensions of a tensor output of the operator.
    STDMETHOD(GetInputTensorDimensionCount)(
        uint32_t inputIndex,
        _Out_ uint32_t* dimensionCount
        ) const noexcept PURE;

    //! Gets the sizes of dimensions of an input tensor of the operator.
    //! Returns an error if the input at the specified index is not a tensor.
    STDMETHOD(GetInputTensorShape)(
        uint32_t inputIndex,
        uint32_t dimensionCount,
        _Out_writes_(dimensionCount) uint32_t* dimensions
        ) const noexcept PURE;

    //! Sets the inferred shape of an output tensor.
    //! Returns an error if the output at the specified index is not a tensor.
    STDMETHOD(SetOutputTensorShape)(
        uint32_t outputIndex, 
        uint32_t dimensionCount, 
        const uint32_t* dimensions
        ) noexcept PURE;
};

//! \interface IMLOperatorTypeInferenceContext
//! \brief Provides information about an operator's usage while type inferrers are being invoked.
interface DECLSPEC_UUID("EC893BB1-F938-427B-8488-C8DCF775F138") DECLSPEC_NOVTABLE
IMLOperatorTypeInferenceContext : public IMLOperatorAttributes 
{
    //! Gets the number of inputs to the operator.
    STDMETHOD_(uint32_t, GetInputCount)() const noexcept PURE;

    //! Gets the number of outputs to the operator.
    STDMETHOD_(uint32_t, GetOutputCount)() const noexcept PURE;

    //! Returns true if an input to the operator is valid.  
    //! This always returns true except for optional inputs.
    STDMETHOD_(bool, IsInputValid)(uint32_t inputIndex) const noexcept PURE;

    //! Returns true if an output to the operator is valid.  
    //! This always returns true except for optional outputs.
    STDMETHOD_(bool, IsOutputValid)(uint32_t outputIndex) const noexcept PURE;

    //! Gets the description of the specified input edge of the operator.
    STDMETHOD(GetInputEdgeDescription)(
        uint32_t inputIndex,
        _Out_ MLOperatorEdgeDescription* edgeDescription
        ) const noexcept PURE;

    //! Sets the inferred type of an output edge.
    STDMETHOD(SetOutputEdgeDescription)(
        uint32_t outputIndex, 
        const MLOperatorEdgeDescription* edgeDescription
        ) const noexcept PURE;
};
 
//! \interface IMLOperatorTypeInferrer
//! \brief Implemented by type inferrers to infer types of an operator's output edges.
//! Type inferrers must be provided when registering schema of custom operators if 
//! the MLOperatorSchemaDescription structure cannot express how output types are 
//! determined.  For example, such as when an attribute of the operator determines 
//! the data type of one of that operator's outputs.
interface DECLSPEC_UUID("781AEB48-9BCB-4797-BF77-8BF455217BEB") DECLSPEC_NOVTABLE
IMLOperatorTypeInferrer : IUnknown
{
    //! Called to infer types of an operator's output edges 
    STDMETHOD(InferOutputTypes)(
        IMLOperatorTypeInferenceContext* context
        ) noexcept PURE;
};

//! \interface IMLOperatorShapeInferrer
//! \brief Implemented by shape inferrers to infer shapes of an operator's 
//! output tensor edges. Shape inferrers may be provided when registering custom 
//! operator kernels to improve performance and to enable the kernel to query 
//! the shape of its output tensors when it is created and computed.  Shape 
//! inferrers may also be provided when registering custom operator schema to
//! improve model validation.
interface DECLSPEC_UUID("540BE5BE-A6C9-40EE-83F6-D2B8B40A7798") DECLSPEC_NOVTABLE
IMLOperatorShapeInferrer : IUnknown
{
    //! Called to infer shapes of an operator's output edges.
    STDMETHOD(InferOutputShapes)(
        IMLOperatorShapeInferenceContext* context
        ) noexcept PURE;
}; 

//! \struct MLOperatorAttribute 
//! \brief Specifies the name and properties of an attribute of a custom operator.
//! This is used when registering custom operator kernels and custom operator schema.
struct MLOperatorAttribute 
{
    //! NULL-terminated UTF-8 string representing the name of the attribute in the 
    //! associated operator type.
    _Field_z_ const char* name;

    //! The type of the attribute in the associated operator type.
    MLOperatorAttributeType type;

    //! Whether the attribute is required in any model using the associated operator type.
    bool required;
};
 
//! \struct MLOperatorAttributeNameValue
//! \brief Specifies the name and value(s) of an attribute of a custom operator.
//! This is used when registering custom operator kernels and custom operator schema.
struct MLOperatorAttributeNameValue 
{
    //! NULL-terminated UTF-8 string representing the name of the attribute in the 
    //! associated operator type.
    _Field_z_ const char* name;

    //! The type of the attribute in the associated operator type.
    MLOperatorAttributeType type;

    //! The number of elements in the attribute value.  This must be one, except for attributes
    //! which are of array types.
    uint32_t valueCount;
 
    union 
    {
        const void* reserved;

        //! 64 bit integer value(s).  Used when the type field is 
        //! MLOperatorAttributeType::Int or MLOperatorAttributeType::IntArray.
        _Field_size_(valueCount) const int64_t* ints;

        //! NULL-terminated UTF-8 string value(s). Used when the type field is 
        //! MLOperatorAttributeType::String or MLOperatorAttributeType::StringArray.
        _Field_size_(valueCount) const char* const* strings;

        //! 32 bit floating point value(s).  Used when the type field is 
        //! MLOperatorAttributeType::Float or MLOperatorAttributeType::FloatArray.
        _Field_size_(valueCount) const float* floats;
    };
};
 
//! \struct MLOperatorSchemaDescription
//! \brief Description of a custom operator schema used to register that schema.
struct MLOperatorSchemaDescription
{
    //! NULL-terminated UTF-8 string representing the name of the operator.
    _Field_z_ const char* name;
 
    //! The operator set version at which this operator was introduced or last changed.
    int32_t operatorSetVersionAtLastChange;
    
    //! An array containing the descriptions of the operator's input edges.
    _Field_size_opt_(inputCount) const MLOperatorSchemaEdgeDescription* inputs;

    //! The number of inputs of the operator.
    uint32_t inputCount;

    //! An array containing the descriptions of the operator's output edges.
    _Field_size_opt_(outputCount) const MLOperatorSchemaEdgeDescription* outputs;

    //! The number of outputs of the operator.
    uint32_t outputCount;
 
    //! An array of type constraints.  Each constraint restricts input and outputs
    //! associated with a type label string to one or more edge types.
    _Field_size_opt_(typeConstraintCount) const MLOperatorEdgeTypeConstraint* typeConstraints;

    //! The number of type constraints provided.
    uint32_t typeConstraintCount;
    
    //! The set of attributes supported by the operator type.
    _Field_size_opt_(attributeCount) const MLOperatorAttribute* attributes;

    //! The number of provided attributes.
    uint32_t attributeCount;
 
    //! The default values of attributes.  These will be applied when the attributes are missing
    //! in a model containing the operator type.
    _Field_size_opt_(defaultAttributeCount) const MLOperatorAttributeNameValue* defaultAttributes;

    //! The number of provided default attribute values.
    uint32_t defaultAttributeCount;
};
 
//! \struct MLOperatorSetId
//! \brief Specifies the identity of an operator set.
struct MLOperatorSetId 
{
    //! The domain of the operator, for example, "ai.onnx.ml", or an empty string
    //! for the ONNX domain.
    _Field_z_ const char* domain;

    //! The version of the operator domain.
    int32_t version;
};
 
//! \enum MLOperatorKernelOptions
//! \brief Specifies options used when registering custom operator kernels.
enum class MLOperatorKernelOptions : uint32_t 
{
    None = 0,
 
    //! Specifies whether the shapes of input tensors are allowed to vary among invocations
    //! of an operator kernel instance.  If this is not set, kernel instances may query input
    //! tensor shapes during creation, and front-load initialization work which depends
    //! on those shapes.  Setting this may improve performance if shapes vary dynamically between
    //! inference operations, and the kernel implementation handles this efficiently. 
    AllowDynamicInputShapes = 1,
};

DEFINE_ENUM_FLAG_OPERATORS(MLOperatorKernelOptions);
 
//! \enum MLOperatorExecutionType
//! \brief Specifies whether a kernel uses the CPU or GPU for computation.
enum class MLOperatorExecutionType : uint32_t 
{
    Undefined = 0,
    Cpu = 1,
    D3D12 = 2
};

//! \struct MLOperatorKernelDescription
//! \brief Description of a custom operator kernel used to register that schema.
struct MLOperatorKernelDescription
{
    //! NULL-terminated UTF-8 string representing the name of the operator's domain.
    _Field_z_ const char* domain;

    //! NULL-terminated UTF-8 string representing the name of the operator.
    _Field_z_ const char* name;
 
    //! The minimum version of the operator sets for which this kernel is valid.  
    //! The maximum version is inferred based on registrations of operator set schema for
    //! subsequent versions of the same domain.
    int32_t minimumOperatorSetVersion;
    
    //! Specifies whether a kernel uses the CPU or GPU for computation.
    MLOperatorExecutionType executionType;
 
    //! An array of type constraints.  Each constraint restricts input and outputs
    //! associated with a type label string to one or more edge types.
    _Field_size_opt_(typeConstraintCount) const MLOperatorEdgeTypeConstraint* typeConstraints;

    //! The number of type constraints provided.
    uint32_t typeConstraintCount;

    //! The default values of attributes.  These will be applied when the attributes are missing
    //! in a model containing the operator type.
    _Field_size_opt_(defaultAttributeCount) const MLOperatorAttributeNameValue* defaultAttributes;

    //! The number of provided default attribute values.
    uint32_t defaultAttributeCount;
    
    //! Options for the kernel which apply to all execution provider types.
    MLOperatorKernelOptions options;
 
    //! Reserved for additional options.  Must be zero.
    uint32_t executionOptions;
};
 
//! \interface IMLOperatorKernelFactory
//! \brief Implemented by the author of a custom operator kernel to create instances of that kernel.
interface DECLSPEC_UUID("EF15AD6F-0DC9-4908-AB35-A575A30DFBF8") DECLSPEC_NOVTABLE
IMLOperatorKernelFactory : IUnknown
{
    //! Creates an instance of the associated operator kernel, given information about the operator's
    //! usage within a model described in the provided context object.
    STDMETHOD(CreateKernel)(
        IMLOperatorKernelCreationContext* context,
        _COM_Outptr_ IMLOperatorKernel** kernel
        ) noexcept PURE;
};
 
//! \interface IMLOperatorRegistry
//! \brief Represents an instance of a registry for custom operator kernel and schema.
//! Custom operators may be used with WinML APIs by returning
//! instances of IMLOperatorRegistry through ILearningModelOperatorProviderNative.
interface DECLSPEC_UUID("2AF9DD2D-B516-4672-9AB5-530C208493AD") DECLSPEC_NOVTABLE
IMLOperatorRegistry : IUnknown
{
    //! Registers a set of custom operator schema comprising an operator set.  Operator sets follow
    //! the ONNX versioning design.  Callers should provide schema for all operators that have changed
    //! between the specified baseline version and the version specified within operatorSetId.  This
    //! prevents older versions of kernels from being used in models which import the newer operator 
    //! set version. A type inferrer must be provided if the MLOperatorSchemaDescription structure
    //! cannot express how output types are determined.  A shape inferrer may optionally be provided
    //! to enable model validation.
    STDMETHOD(RegisterOperatorSetSchema)(
        const MLOperatorSetId* operatorSetId,
        int32_t baselineVersion,
        _In_reads_opt_(schemaCount) const MLOperatorSchemaDescription* const* schema,
        uint32_t schemaCount,
        _In_opt_ IMLOperatorTypeInferrer* typeInferrer,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer
        ) const noexcept PURE;

    //! Registers a custom operator kernel.  
    //! A shape inferrer may optionally be provided.  This may improve performance and enables
    //! the kernel to query the shape of its output tensors when it is created and computed.
    STDMETHOD(RegisterOperatorKernel)(
        const MLOperatorKernelDescription* operatorKernel,
        IMLOperatorKernelFactory* operatorKernelFactory,
        _In_opt_ IMLOperatorShapeInferrer* shapeInferrer
        ) const noexcept PURE;
};
 
extern "C"
{
    //! \fn MLCreateOperatorRegistry
    //! Creates an instance of IMLOperatorRegistry which may be used to register custom
    //! operator kernel and custom operator schema. 
    HRESULT WINAPI MLCreateOperatorRegistry(_COM_Outptr_ IMLOperatorRegistry** registry);
}

#endif /* WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) */

#endif /* defined(__cplusplus) */
#endif /* defined(_MSC_VER) && (_MSC_VER >= 1700) */