using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
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
                .Append(_mlContext.Transforms.ResizeImages(
                    resizing: ImageResizingEstimator.ResizingKind.Fill,
                    outputColumnName: onnxModel.ModelInput,
                    imageWidth: ImageSettings.imageWidth,
                    imageHeight: ImageSettings.imageHeight,
                    inputColumnName: nameof(ImageInputData.Image)))
                .Append(_mlContext.Transforms.ExtractPixels(
                        inputColumnName: onnxModel.ModelInput, 
                        outputColumnName: onnxModel.ModelInput))
                .Append(_mlContext.Transforms.ApplyOnnxModel(
                    modelFile: onnxModel.ModelPath, 
                    outputColumnName: onnxModel.ModelOutput, 
                    inputColumnName: onnxModel.ModelInput))
                .Append(_mlContext.Transforms.CopyColumns(inputColumnName: onnxModel.ModelOutput,
                    outputColumnName: nameof(IOnnxObjectPrediction.PredictedLabels)));

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        public PredictionEngine<ImageInputData, T> GetMlNetPredictionEngine<T>()
            where T : class, IOnnxObjectPrediction, new()
        {
            return _mlContext.Model.CreatePredictionEngine<ImageInputData, T>(_mlModel);
        }

        public void SaveMlNetModel(string mlnetModelFilePath)
        {
            // Save/persist the model to a .ZIP file to be loaded by the PredictionEnginePool
            _mlContext.Model.Save(_mlModel, null, mlnetModelFilePath);
        }
    }
}
