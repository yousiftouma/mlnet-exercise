using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace OnnxObjectDetection.ML.DataModels
{
    public struct ImageSettings
    {
        public const int imageHeight = int.MinValue; //TODO update these
        public const int imageWidth = int.MinValue;
    }

    public class ImageInputData
    {
        //TODO annotate this
        public Bitmap Image { get; set; }
    }
}
