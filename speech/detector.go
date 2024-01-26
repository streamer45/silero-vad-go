package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"log/slog"
	"runtime"
	"unsafe"
)

const (
	hcLen = 2 * 1 * 64
)

type DetectorConfig struct {
	// The path to the ONNX Silero VAD model file to load.
	ModelPath string
	// The sampling rate of the input audio samples. Supported values are 8000 and 16000.
	SampleRate int
	// The number of samples to process at each infer.
	WindowSize int
	// The probability threshold above which we detect speech. A good default is 0.5.
	Threshold float32
	// The duration of silence to wait for each speech segment before separating it.
	MinSilenceDurationMs int
	// The duration of speech to wait for before starting a speech segment.
	MinSpeechDurationMs int
	// The padding to add to speech segments to avoid aggressive cutting.
	SpeechPadMs int
	// The padding to add to silence segments to avoid aggressive zeroing.
	SilencePadMs int
}

func (c DetectorConfig) IsValid() error {
	if c.ModelPath == "" {
		return fmt.Errorf("invalid ModelPath: should not be empty")
	}

	if c.SampleRate != 8000 && c.SampleRate != 16000 {
		return fmt.Errorf("invalid SampleRate: valid values are 8000 and 16000")
	}

	if (c.SampleRate == 16000 && c.WindowSize != 512 && c.WindowSize != 1024 && c.WindowSize != 1536) ||
		(c.SampleRate == 8000 && c.WindowSize != 256 && c.WindowSize != 512 && c.WindowSize != 768) {
		return fmt.Errorf("invalid WindowSize: valid values are 512, 1024, 1536 for 16000 sample rate and 256, 512, 768 for 8000 sample rate")
	}

	if c.Threshold <= 0 || c.Threshold >= 1 {
		return fmt.Errorf("invalid Threshold: should be in range (0, 1)")
	}

	if c.MinSilenceDurationMs < 0 {
		return fmt.Errorf("invalid MinSilenceDurationMs: should be a positive number")
	}

	if c.MinSpeechDurationMs < 0 {
		return fmt.Errorf("invalid MinSpeechDurationMs: should be a positive number")
	}

	if c.SpeechPadMs < 0 {
		return fmt.Errorf("invalid SpeechPadMs: should be a positive number")
	}

	if c.SilencePadMs < 0 {
		return fmt.Errorf("invalid SilencePadMs: should be a positive number")
	}

	return nil
}

type Detector struct {
	api         *C.OrtApi
	env         *C.OrtEnv
	sessionOpts *C.OrtSessionOptions
	session     *C.OrtSession
	memoryInfo  *C.OrtMemoryInfo
	cStrings    map[string]*C.char

	cfg DetectorConfig

	h [hcLen]float32
	c [hcLen]float32

	currSample int
	triggered  bool
	tempEnd    int
}

func NewDetector(cfg DetectorConfig) (*Detector, error) {
	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	sd := Detector{
		cfg:      cfg,
		cStrings: map[string]*C.char{},
	}

	sd.api = C.OrtGetApi()
	if sd.api == nil {
		return nil, fmt.Errorf("failed to get API")
	}

	sd.cStrings["loggerName"] = C.CString("vad")
	status := C.OrtApiCreateEnv(sd.api, C.ORT_LOGGING_LEVEL_WARNING, sd.cStrings["loggerName"], &sd.env)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create env: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiCreateSessionOptions(sd.api, &sd.sessionOpts)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session options: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetIntraOpNumThreads(sd.api, sd.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set intra threads: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetInterOpNumThreads(sd.api, sd.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set inter threads: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiSetSessionGraphOptimizationLevel(sd.api, sd.sessionOpts, C.ORT_ENABLE_ALL)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set session graph optimization level: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	sd.cStrings["modelPath"] = C.CString(sd.cfg.ModelPath)
	status = C.OrtApiCreateSession(sd.api, sd.env, sd.cStrings["modelPath"], sd.sessionOpts, &sd.session)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	status = C.OrtApiCreateCpuMemoryInfo(sd.api, C.OrtArenaAllocator, C.OrtMemTypeDefault, &sd.memoryInfo)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create memory info: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}

	sd.cStrings["input"] = C.CString("input")
	sd.cStrings["sr"] = C.CString("sr")
	sd.cStrings["h"] = C.CString("h")
	sd.cStrings["c"] = C.CString("c")
	sd.cStrings["output"] = C.CString("output")
	sd.cStrings["hn"] = C.CString("hn")
	sd.cStrings["cn"] = C.CString("cn")

	return &sd, nil
}

func (sd *Detector) infer(pcm []float32) (float32, error) {
	// Create tensors
	var pcmValue *C.OrtValue

	var pcmInputDims []any = []C.long{
		1,
		C.longlong(len(pcm)),
	}
	if runtime.GOOS == "darwin" {
		pcmInputDims = []C.longlong{
			1,
			C.longlong(len(pcm)),
		}
	}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&pcm[0]), C.size_t(len(pcm)*4), &pcmInputDims[0], C.size_t(len(pcmInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pcmValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, pcmValue)

	var rateValue *C.OrtValue

	var rateInputDims []any = []C.long{1}
	if runtime.GOOS == "darwin" {
		rateInputDims = []C.longlong{1}
	}

	rate := []C.int64_t{C.int64_t(sd.cfg.SampleRate)}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(sd.api, sd.memoryInfo, unsafe.Pointer(&rate[0]), C.size_t(8), &rateInputDims[0], C.size_t(len(rateInputDims)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &rateValue)
	defer C.OrtApiReleaseStatus(sd.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create value: %s", C.GoString(C.OrtApiGetErrorMessage(sd.api, status)))
	}
	defer C.OrtApiReleaseValue(sd.api, rateValue)

	var hcNodeInputDims []any = []C.long{2, 1, 64}
	if runtime.GOOS == "darwin" {
		hcNodeInputDims = []C.longlong{2, 1, 64}
	}

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

// Segment contains timing information of a speech segment.
type Segment struct {
	// The relative timestamp in seconds of when a speech segment begins.
	SpeechStartAt float64
	// The relative timestamp in seconds of when a speech segment ends.
	SpeechEndAt float64
}

func (sd *Detector) Detect(pcm []float32) ([]Segment, error) {
	if sd == nil {
		return nil, fmt.Errorf("invalid nil detector")
	}

	if len(pcm) < sd.cfg.WindowSize {
		return nil, fmt.Errorf("not enough samples")
	}

	slog.Debug("starting speech detection", slog.Int("samplesLen", len(pcm)))

	minSilenceSamples := sd.cfg.MinSilenceDurationMs * sd.cfg.SampleRate / 1000
	speechPadSamples := sd.cfg.SpeechPadMs * sd.cfg.SampleRate / 1000

	var segments []Segment
	for i := 0; i < len(pcm)-sd.cfg.WindowSize; i += sd.cfg.WindowSize {
		speechProb, err := sd.infer(pcm[i : i+sd.cfg.WindowSize])
		if err != nil {
			return nil, fmt.Errorf("infer failed: %w", err)
		}

		sd.currSample += sd.cfg.WindowSize

		if speechProb >= sd.cfg.Threshold && sd.tempEnd != 0 {
			sd.tempEnd = 0
		}

		if speechProb >= sd.cfg.Threshold && !sd.triggered {
			sd.triggered = true
			speechStartAt := (float64(sd.currSample-sd.cfg.WindowSize-speechPadSamples) / float64(sd.cfg.SampleRate))

			// We clamp at zero since due to padding the starting position could be negative.
			if speechStartAt < 0 {
				speechStartAt = 0
			}

			slog.Debug("speech start", slog.Float64("startAt", speechStartAt))
			segments = append(segments, Segment{
				SpeechStartAt: speechStartAt,
			})
		}

		if speechProb < (sd.cfg.Threshold-0.15) && sd.triggered {
			if sd.tempEnd == 0 {
				sd.tempEnd = sd.currSample
			}

			// Not enough silence yet to split, we continue.
			if sd.currSample-sd.tempEnd < minSilenceSamples {
				continue
			}

			speechEndAt := (float64(sd.tempEnd+speechPadSamples) / float64(sd.cfg.SampleRate))
			sd.tempEnd = 0
			sd.triggered = false
			slog.Debug("speech end", slog.Float64("endAt", speechEndAt))

			if len(segments) < 1 {
				return nil, fmt.Errorf("unexpected speech end")
			}

			segments[len(segments)-1].SpeechEndAt = speechEndAt
		}
	}

	slog.Debug("speech detection done", slog.Int("segmentsLen", len(segments)))

	return segments, nil
}

type RealtimeSegment struct {
	Start   int
	End     int
	Silence bool
}

// DetectRealtime returns the `cleaned` pcm (silence is fully zeroed out to improve transcriber accuracy),
// and the voice and silence segments are detected and recorded in `segments`.
func (sd *Detector) DetectRealtime(pcm []float32) (cleaned []float32, segments []RealtimeSegment, retErr error) {
	if sd == nil {
		retErr = fmt.Errorf("invalid nil detector")
		return
	}

	if len(pcm) < sd.cfg.WindowSize {
		retErr = fmt.Errorf("not enough samples")
		return
	}

	cleaned = make([]float32, 0, len(pcm))
	cleaned = append(cleaned, pcm...)

	minSilenceSamples := sd.cfg.MinSilenceDurationMs * sd.cfg.SampleRate / 1000
	minSpeechSamples := sd.cfg.MinSpeechDurationMs * sd.cfg.SampleRate / 1000
	speechPadSamples := sd.cfg.SpeechPadMs * sd.cfg.SampleRate / 1000
	tempEndSpeech := 0
	triggeredSpeech := false
	currStartSpeech := 0
	tempStartSpeech := 0

	currStartSilence := 0

	for i := 0; i <= len(cleaned)-sd.cfg.WindowSize; i += sd.cfg.WindowSize {
		speechProb, err := sd.infer(cleaned[i : i+sd.cfg.WindowSize])
		if err != nil {
			retErr = fmt.Errorf("infer failed: %w", err)
			return
		}

		currSampleStart := i
		currSampleEnd := i + sd.cfg.WindowSize

		if speechProb >= sd.cfg.Threshold {
			if tempStartSpeech == 0 {
				tempStartSpeech = currSampleStart
			}
			// Not enough speech samples yet to declare this a speech segment, we continue.
			if currSampleEnd-tempStartSpeech < minSpeechSamples {
				continue
			}

			if tempEndSpeech != 0 {
				tempEndSpeech = 0
			}

			if !triggeredSpeech {
				triggeredSpeech = true
				startSpeech := tempStartSpeech - speechPadSamples
				startSpeech = max(startSpeech, 0)
				currStartSpeech = startSpeech
				endSilence := startSpeech

				// only record silence if it reached the threshold
				if endSilence-currStartSilence >= minSilenceSamples {
					// zero out silence, and record it as a segment
					for j := currStartSilence; j < endSilence; j++ {
						cleaned[j] = 0
					}
					segments = append(segments, RealtimeSegment{
						Start:   currStartSilence,
						End:     endSilence,
						Silence: true,
					})
				}

				//slog.Debug("speech start", slog.Float64("startAt", speechStartAt))
			}
		}

		if speechProb < (sd.cfg.Threshold-0.15) && triggeredSpeech {
			if tempEndSpeech == 0 {
				tempEndSpeech = currSampleStart
			}

			// Not enough silence yet to split, we continue.
			if currSampleEnd-tempEndSpeech < minSilenceSamples {
				continue
			}

			// enough silence
			triggeredSpeech = false
			endSample := tempEndSpeech + speechPadSamples
			endSample = min(endSample, len(cleaned))
			currEnd := endSample
			currStartSilence = endSample
			tempStartSpeech = 0
			tempEndSpeech = 0
			//slog.Debug("speech end", slog.Float64("endAt", speechEndAt))

			segments = append(segments, RealtimeSegment{
				Start: currStartSpeech,
				End:   currEnd,
			})
		}
	}

	// close off current sample
	if triggeredSpeech {
		segments = append(segments, RealtimeSegment{
			Start: currStartSpeech,
			End:   len(cleaned),
		})
	} else {
		// zero out silence, and record it as a segment
		for j := currStartSilence; j < len(cleaned); j++ {
			cleaned[j] = 0
		}
		segments = append(segments, RealtimeSegment{
			Start:   currStartSilence,
			End:     len(cleaned),
			Silence: true,
		})
	}

	return
}

func (sd *Detector) Reset() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	sd.currSample = 0
	sd.triggered = false
	sd.tempEnd = 0
	for i := 0; i < hcLen; i++ {
		sd.h[i] = 0
		sd.c[i] = 0
	}

	return nil
}

func (sd *Detector) Destroy() error {
	if sd == nil {
		return fmt.Errorf("invalid nil detector")
	}

	C.OrtApiReleaseMemoryInfo(sd.api, sd.memoryInfo)
	C.OrtApiReleaseSession(sd.api, sd.session)
	C.OrtApiReleaseSessionOptions(sd.api, sd.sessionOpts)
	C.OrtApiReleaseEnv(sd.api, sd.env)
	for _, ptr := range sd.cStrings {
		C.free(unsafe.Pointer(ptr))
	}

	return nil
}
