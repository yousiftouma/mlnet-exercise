using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using OnnxObjectDetection.ML.DataModels;

namespace OnnxObjectDetection.Cheating
{
    public static class Cheats
    {
        public static ImageResizingEstimator ResizeImage(MLContext context, IOnnxModel model)
        {
            return context.Transforms.ResizeImages(
                inputColumnName: nameof(ImageInputData.Image),
                outputColumnName: model.ModelInput,
                resizing: ImageResizingEstimator.ResizingKind.Fill,
                imageWidth: ImageSettings.imageWidth,
                imageHeight: ImageSettings.imageHeight);
        }

        public static ImagePixelExtractingEstimator ExtractPixels(MLContext context, IOnnxModel model)
        {
            return context.Transforms.ExtractPixels(
                inputColumnName: model.ModelInput,
                outputColumnName: model.ModelInput);
        }

        public static OnnxScoringEstimator ApplyOnnxModel(MLContext context, IOnnxModel model)
        {
            return context.Transforms.ApplyOnnxModel(
                inputColumnName: model.ModelInput,
                outputColumnName: model.ModelOutput,
                modelFile: model.ModelPath);
        }

        public static ColumnCopyingEstimator CopyColumns(MLContext context, IOnnxModel model)
        {
            return context.Transforms.CopyColumns(
                inputColumnName: model.ModelOutput,
                outputColumnName: nameof(IOnnxObjectPrediction.PredictedLabels));
        }
    }
}
