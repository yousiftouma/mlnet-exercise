using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using Microsoft.ML.Transforms.Onnx;
using OnnxObjectDetection.Cheating;
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
            // This is required to understand the shape of the input
            var dataView = _mlContext.Data.LoadFromEnumerable(new List<ImageInputData>());

            var pipeline = new EstimatorChain<ITransformer>();

            //TODO append the estimators to the pipeline in the correct order

            var mlNetModel = pipeline.Fit(dataView);

            return mlNetModel;
        }

        private ImageResizingEstimator ResizeImage(IOnnxModel onnxModel)
        {
            //TODO Return an estimator that resizes a Bitmap
            // (note: cheating possible through Cheats.ResizeImage)
            throw new NotImplementedException();
        }

        private ImagePixelExtractingEstimator ExtractPixels(IOnnxModel onnxModel)
        {
            //TODO Return an estimator that extracts pixel values from a Bitmap
            // (note: cheating possible through Cheats.ExtractPixels)
            throw new NotImplementedException();
        }

        private OnnxScoringEstimator ApplyOnnxModel(IOnnxModel onnxModel)
        {
            //TODO Return an estimator that runs a float vector through an Onnx model
            // (note: cheating possible through Cheats.ApplyOnnxModel)
            throw new NotImplementedException();
        }

        private ColumnCopyingEstimator CopyColumns(IOnnxModel onnxModel)
        {
            //TODO Return an estimator that copies values
            // (note: cheating possible through Cheats.CopyColumns)
            throw new NotImplementedException();
        }

        public PredictionEngine<ImageInputData, T> GetMlNetPredictionEngine<T>()
            where T : class, IOnnxObjectPrediction, new()
        {
            //TODO create and return a prediction engine 
            // the prediction engine should run objects of type ImageInputData into the pipeline
            // and store the output in objects of the generic type T (what are the constraints on it?)
            throw new NotImplementedException();
        }

        public void SaveMlNetModel(string mlnetModelFilePath)
        {
            // not needed yet
            throw new NotImplementedException();
        }
    }
}
