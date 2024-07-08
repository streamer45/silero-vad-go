package speech

import (
	"encoding/binary"
	"log/slog"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDetectorConfigIsValid(t *testing.T) {
	tcs := []struct {
		name string
		cfg  DetectorConfig
		err  string
	}{
		{
			name: "missing ModelPath",
			cfg: DetectorConfig{
				ModelPath: "",
			},
			err: "invalid ModelPath: should not be empty",
		},
		{
			name: "invalid SampleRate",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 48000,
			},
			err: "invalid SampleRate: valid values are 8000 and 16000",
		},
		{
			name: "invalid Threshold",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 16000,
				Threshold:  0,
			},
			err: "invalid Threshold: should be in range (0, 1)",
		},
		{
			name: "invalid MinSilenceDurationMs",
			cfg: DetectorConfig{
				ModelPath:            "../testfiles/silero_vad.onnx",
				SampleRate:           16000,
				Threshold:            0.5,
				MinSilenceDurationMs: -1,
			},
			err: "invalid MinSilenceDurationMs: should be a positive number",
		},
		{
			name: "invalid SpeechPadMs",
			cfg: DetectorConfig{
				ModelPath:   "../testfiles/silero_vad.onnx",
				SampleRate:  16000,
				Threshold:   0.5,
				SpeechPadMs: -1,
			},
			err: "invalid SpeechPadMs: should be a positive number",
		},
		{
			name: "valid",
			cfg: DetectorConfig{
				ModelPath:  "../testfiles/silero_vad.onnx",
				SampleRate: 16000,
				Threshold:  0.5,
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.IsValid()
			if tc.err != "" {
				require.EqualError(t, err, tc.err)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestNewDetector(t *testing.T) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	sd, err := NewDetector(cfg)
	require.NoError(t, err)
	require.NotNil(t, sd)

	err = sd.Destroy()
	require.NoError(t, err)
}

func TestSpeechDetection(t *testing.T) {
	cfg := DetectorConfig{
		ModelPath:  "../testfiles/silero_vad.onnx",
		SampleRate: 16000,
		Threshold:  0.5,
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		AddSource: true,
		Level:     slog.LevelDebug,
	}))
	slog.SetDefault(logger)

	sd, err := NewDetector(cfg)
	require.NoError(t, err)
	require.NotNil(t, sd)
	defer func() {
		require.NoError(t, sd.Destroy())
	}()

	data, err := os.ReadFile("../testfiles/samples.pcm")
	require.NoError(t, err)

	samples := make([]float32, 0, len(data)/4)
	for i := 0; i < len(data); i += 4 {
		samples = append(samples, math.Float32frombits(binary.LittleEndian.Uint32(data[i:i+4])))
	}

	t.Run("detect", func(t *testing.T) {
		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		require.Equal(t, []Segment{
			{
				SpeechStartAt: 1.088,
				SpeechEndAt:   1.632,
			},
			{
				SpeechStartAt: 2.912,
				SpeechEndAt:   3.264,
			},
			{
				SpeechStartAt: 4.448,
				SpeechEndAt:   0,
			},
		}, segments)
	})

	t.Run("reset", func(t *testing.T) {
		err = sd.Reset()
		require.NoError(t, err)

		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		require.Equal(t, []Segment{
			{
				SpeechStartAt: 1.088,
				SpeechEndAt:   1.632,
			},
			{
				SpeechStartAt: 2.912,
				SpeechEndAt:   3.264,
			},
			{
				SpeechStartAt: 4.448,
				SpeechEndAt:   0,
			},
		}, segments)
	})

	t.Run("speech padding", func(t *testing.T) {
		cfg.SpeechPadMs = 10
		sd, err := NewDetector(cfg)
		require.NoError(t, err)
		require.NotNil(t, sd)
		defer func() {
			require.NoError(t, sd.Destroy())
		}()

		segments, err := sd.Detect(samples)
		require.NoError(t, err)
		require.NotEmpty(t, segments)
		require.Equal(t, []Segment{
			{
				SpeechStartAt: 1.088 - 0.01,
				SpeechEndAt:   1.632 + 0.01,
			},
			{
				SpeechStartAt: 2.912 - 0.01,
				SpeechEndAt:   3.264 + 0.01,
			},
			{
				SpeechStartAt: 4.448 - 0.01,
				SpeechEndAt:   0,
			},
		}, segments)
	})
}
