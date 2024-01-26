//go:build darwin

package speech

import "C"
import (
	"fmt"
	"unsafe"
)

func (sd *Detector) infer(pcm []float32) (float32, error) {
	// Create tensors
	var pcmValue *C.OrtValue
	pcmInputDims := []C.longlong{
		1,
		C.longlong(len(pcm)),
	}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&pcm[0]), C.size_t(len(pcm)*4), &pcmInputDims[0], C.size_t(len(pcmInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pcmValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, pcmValue)

	var rateValue *C.OrtValue
	rateInputDims := []C.longlong{1}
	rate := []C.int64_t{C.int64_t(sd.cfg.SampleRate)}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&rate[0]), C.size_t(8), &rateInputDims[0], C.size_t(len(rateInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &rateValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, rateValue)

	hcNodeInputDims := []C.longlong{2, 1, 64}

	var hValue *C.OrtValue
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.h[0]), C.size_t(hcLen*4), &hcNodeInputDims[0], C.size_t(len(hcNodeInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &hValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, hValue)

	var cValue *C.OrtValue
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&sd.c[0]), C.size_t(hcLen*4), &hcNodeInputDims[0], C.size_t(len(hcNodeInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &cValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, cValue)

	// Run inference
	inputs := []*C.OrtValue{pcmValue, rateValue, hValue, cValue}
	outputs := []*C.OrtValue{nil, nil, nil}

	inputNames := []*C.char{
		sd.cStrings["input"],
		sd.cStrings["sr"],
		sd.cStrings["h"],
		sd.cStrings["c"],
	}
	outputNames := []*C.char{
		sd.cStrings["output"],
		sd.cStrings["hn"],
		sd.cStrings["cn"],
	}
	status = C.OrtApiRun(sd.api, sd.session, nil, &inputNames[0], &inputs[0], C.size_t(len(inputNames)), &outputNames[0], C.size_t(len(outputNames)), &outputs[0])
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to run: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	// Get output values from tensor data
	var prob unsafe.Pointer
	var hn unsafe.Pointer
	var cn unsafe.Pointer

	status = C.OrtApiGetTensorMutableData(sd.api, outputs[0], &prob)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiGetTensorMutableData(sd.api, outputs[1], &hn)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiGetTensorMutableData(sd.api, outputs[2], &cn)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	C.memcpy(unsafe.Pointer(&sd.h[0]), hn, hcLen*4)
	C.memcpy(unsafe.Pointer(&sd.c[0]), cn, hcLen*4)

	C.OrtApiReleaseValue(sd.api, outputs[0])
	C.OrtApiReleaseValue(sd.api, outputs[1])
	C.OrtApiReleaseValue(sd.api, outputs[2])

	// Return speech probability
	return *(*float32)(prob), nil
}
