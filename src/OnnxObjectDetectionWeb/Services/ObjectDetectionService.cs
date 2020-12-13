using Microsoft.Extensions.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using OnnxObjectDetection;
using OnnxObjectDetection.Cheating;
using OnnxObjectDetection.ML;
using OnnxObjectDetection.ML.DataModels;
using OnnxObjectDetectionWeb.Utilities;

namespace OnnxObjectDetectionWeb.Services
{
    public interface IObjectDetectionService
    {
        /// <summary>
        /// Detect objects in an image, returning bounding boxes of the most probable objects.
        /// </summary>
        /// <param name="imageInputData">
        /// The input image object.
        /// </param>
        /// <returns>
        /// A list of bounding boxes with the most probable objects.
        /// </returns>
        IEnumerable<BoundingBox> DetectObjectsUsingModel(ImageInputData imageInputData);
        /// <summary>
        /// Draws bounding boxes on image.
        /// </summary>
        /// <param name="imageFilePath">
        /// Path to image to draw on.
        /// </param>
        /// <param name="boundingBoxes">
        /// The boxes to draw.
        /// </param>
        /// <returns>
        /// The image, now with bounding boxes on it.
        /// </returns>
        Image DrawBoundingBox(string imageFilePath, IEnumerable<BoundingBox> boundingBoxes);
    }

    public class ObjectDetectionService : IObjectDetectionService
    {
        private readonly IOnnxOutputParser _outputParser;

        public ObjectDetectionService()
        {
            // we create an output parser using the definition of the onnx model, so no need to include a path to an actual onnx model.
            _outputParser = new OnnxOutputParser(new TinyYoloModel(null));
        }

        public IEnumerable<BoundingBox> DetectObjectsUsingModel(ImageInputData imageInputData)
        {
            //TODO Create a transformer pipeline (Hint: there are CommonHelpers to help get a correct path to the onnx model)

            //TODO Get a prediction engine and use it

            //TODO Parse output from the model using the output parser and generate a list of bounding boxes

            //TODO filter the bounding boxes with some limit and threshold for inclusion

            // NOTE: you can use cheat for this whole method (Cheating.DetectObjectsUsingModel)

            throw new NotImplementedException();
        }

        public Image DrawBoundingBox(string imageFilePath, IEnumerable<BoundingBox> boundingBoxes)
        {
            var image = Image.FromFile(imageFilePath);
            var originalHeight = image.Height;
            var originalWidth = image.Width;
            foreach (var box in boundingBoxes)
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
