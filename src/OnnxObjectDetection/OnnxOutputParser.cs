using System;
using System.Collections.Generic;
using OnnxObjectDetection.ML.DataModels;
using OnnxObjectDetection.Cheating;

namespace OnnxObjectDetection
{
    //TODO implement this class
    // Cheat available: You can just remove all the code in the class and extend Cheating.OnnxOutputParserCheat if you want.
    public class OnnxOutputParser : IOnnxOutputParser
    {
        // Labels corresponding to the classes the onnx model can predict. For example, the 
        // Tiny YOLOv2 model included with this sample is trained to predict 20 different classes.
        private readonly string[] _classLabels;

        // Predetermined anchor offsets for the bounding boxes in a cell.
        private readonly (float x, float y)[] _boxAnchors;

        public OnnxOutputParser(IOnnxModel onnxModel)
        {
            _classLabels = onnxModel.Labels;
            _boxAnchors = onnxModel.Anchors;
        }

        public IReadOnlyCollection<BoundingBox> ParseOutputs(float[] modelOutput, float probabilityThreshold = 0.3f)
        {
            throw new NotImplementedException();
        }

        public IEnumerable<BoundingBox> FilterBoundingBoxes(IReadOnlyCollection<BoundingBox> boxes, int limit, float iouThreshold)
        {
            throw new NotImplementedException();
        }
    }
}