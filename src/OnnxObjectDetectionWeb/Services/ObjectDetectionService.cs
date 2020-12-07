using Microsoft.Extensions.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using OnnxObjectDetection;

namespace OnnxObjectDetectionWeb.Services
{
    public interface IObjectDetectionService
    {
        void DetectObjectsUsingModel(ImageInputData imageInputData);
        Image DrawBoundingBox(string imageFilePath);
    }

    public class ObjectDetectionService : IObjectDetectionService
    {
        List<BoundingBox> _filteredBoxes;
        private readonly OnnxOutputParser _outputParser = new OnnxOutputParser(new TinyYoloModel(null));
        private readonly PredictionEnginePool<ImageInputData, TinyYoloPrediction> _predictionEngine;

        public ObjectDetectionService(PredictionEnginePool<ImageInputData, TinyYoloPrediction> predictionEngine)
        {
            _predictionEngine = predictionEngine;
        }

        public void DetectObjectsUsingModel(ImageInputData imageInputData)
        {
            var probabilities = _predictionEngine.Predict(imageInputData).PredictedLabels;
            var boundingBoxes = _outputParser.ParseOutputs(probabilities);
            _filteredBoxes = OnnxOutputParser.FilterBoundingBoxes(boundingBoxes, 5, .5F);
        }

        public Image DrawBoundingBox(string imageFilePath)
        {
            var image = Image.FromFile(imageFilePath);
            var originalHeight = image.Height;
            var originalWidth = image.Width;
            foreach (var box in _filteredBoxes)
            {
                // process output boxes
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = (uint)Math.Min(originalWidth - x, box.Dimensions.Width);
                var height = (uint)Math.Min(originalHeight - y, box.Dimensions.Height);

                // fit to current image size
                x = (uint)originalWidth * x / ImageSettings.imageWidth;
                y = (uint)originalHeight * y / ImageSettings.imageHeight;
                width = (uint)originalWidth * width / ImageSettings.imageWidth;
                height = (uint)originalHeight * height / ImageSettings.imageHeight;

                using (var thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    var drawFont = new Font("Arial", 12, FontStyle.Bold);
                    var size = thumbnailGraphic.MeasureString(box.Description, drawFont);
                    var fontBrush = new SolidBrush(Color.Black);
                    var atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    // Define BoundingBox options
                    var pen = new Pen(box.BoxColor, 3.2f);
                    var colorBrush = new SolidBrush(box.BoxColor);

                    // Draw text on image 
                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    thumbnailGraphic.DrawString(box.Description, drawFont, fontBrush, atPoint);

                    // Draw bounding box on image
                    thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                }
            }
            return image;
        }
    }
}
