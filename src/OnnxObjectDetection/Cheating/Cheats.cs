using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using OnnxObjectDetection.ML;
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

        public static IEnumerable<BoundingBox> DetectObjectsUsingModel(ImageInputData imageInputData, IOnnxOutputParser outputParser)
        {
            var transformer = new OnnxModelConfigurator(new TinyYoloModel(GetAbsolutePath("ML/OnnxModels/TinyYolo2_model.onnx")));
            var predictor = transformer.GetMlNetPredictionEngine<TinyYoloPrediction>();

            var probabilities = predictor.Predict(imageInputData).PredictedLabels;

            var boundingBoxes = outputParser.ParseOutputs(probabilities);
            return outputParser.FilterBoundingBoxes(boundingBoxes, 5, .5F);
        }

        private static string GetAbsolutePath(string relativePath)
        {
            var assembly = Assembly.GetExecutingAssembly();
            var dataRoot = new FileInfo(assembly.Location);
            var assemblyFolderPath = dataRoot.Directory.FullName;

            var fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}
