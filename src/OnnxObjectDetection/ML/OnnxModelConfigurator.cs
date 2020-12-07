using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using OnnxObjectDetection.ML.DataModels;

namespace OnnxObjectDetection.ML
{
    public class OnnxModelConfigurator
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _mlModel;

        public OnnxModelConfigurator(IOnnxModel onnxModel)
        {
            _mlContext = new MLContext();
            // Model creation and pipeline definition for images needs to run just once,
            // so calling it from the constructor:
            _mlModel = SetupMlNetModel(onnxModel);
        }

        private ITransformer SetupMlNetModel(IOnnxModel onnxModel)
        {
            var dataView = _mlContext.Data.LoadFromEnumerable(new List<ImageInputData>());

            var pipeline = new EstimatorChain<ITransformer>()
                .Append(ResizeImage(onnxModel))
                .Append(ExtractPixels(onnxModel))
                .Append(ApplyOnnxModel(onnxModel))
                .Append(CopyColumns(onnxModel));

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        private ImageResizingEstimator ResizeImage(IOnnxModel onnxModel)
        {
            return _mlContext.Transforms.ResizeImages(
                inputColumnName: nameof(ImageInputData.Image),
                outputColumnName: onnxModel.ModelInput,
                resizing: ImageResizingEstimator.ResizingKind.Fill,
                imageWidth: ImageSettings.imageWidth,
                imageHeight: ImageSettings.imageHeight);
        }

        private ImagePixelExtractingEstimator ExtractPixels(IOnnxModel onnxModel)
        {
            return _mlContext.Transforms.ExtractPixels(
                inputColumnName: onnxModel.ModelInput, 
                outputColumnName: onnxModel.ModelInput);
        }

        private OnnxScoringEstimator ApplyOnnxModel(IOnnxModel onnxModel)
        {
            return _mlContext.Transforms.ApplyOnnxModel(
                inputColumnName: onnxModel.ModelInput,
                outputColumnName: onnxModel.ModelOutput,
                modelFile: onnxModel.ModelPath);
        }

        private ColumnCopyingEstimator CopyColumns(IOnnxModel onnxModel)
        {
            return _mlContext.Transforms.CopyColumns(inputColumnName: onnxModel.ModelOutput,
                outputColumnName: nameof(IOnnxObjectPrediction.PredictedLabels));
        }

        public PredictionEngine<ImageInputData, T> GetMlNetPredictionEngine<T>()
            where T : class, IOnnxObjectPrediction, new()
        {
            return _mlContext.Model.CreatePredictionEngine<ImageInputData, T>(_mlModel);
        }

        public void SaveMlNetModel(string mlnetModelFilePath)
        {
            //TODO Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
            throw new NotImplementedException();
        }
    }
}
