namespace OnnxObjectDetection.ML.DataModels
{
    public class TinyYoloPrediction : IOnnxObjectPrediction
    {
        public float[] PredictedLabels { get; set; }
    }
}
