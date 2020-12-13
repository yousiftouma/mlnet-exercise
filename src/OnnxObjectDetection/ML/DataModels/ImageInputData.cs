using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace OnnxObjectDetection.ML.DataModels
{
    public struct ImageSettings
    {
        public const int imageHeight = 0; //TODO update these
        public const int imageWidth = 0;
    }

    public class ImageInputData
    {
        //TODO annotate this
        public Bitmap Image { get; set; }
    }
}
