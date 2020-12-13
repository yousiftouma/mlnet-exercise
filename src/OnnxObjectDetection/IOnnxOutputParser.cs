using System.Collections.Generic;

namespace OnnxObjectDetection
{
    public interface IOnnxOutputParser
    {
        /// <summary>
        /// Parses output from Onnx model into a usable collection of bounding boxes.
        /// </summary>
        /// <param name="modelOutput">
        /// Output from Onnx, a float tensor of form 125x13x13, flattened into a 1d float array.
        /// </param>
        /// <param name="probabilityThreshold">
        /// The minimum probability for a bounding box to be included.
        /// </param>
        IReadOnlyCollection<BoundingBox> ParseOutputs(float[] modelOutput, float probabilityThreshold = .3f);

        /// <summary>
        /// Filters a set of bounding boxes.
        /// </summary>
        /// <param name="boxes">
        /// The boxes to filter.
        /// </param>
        /// <param name="limit">
        /// Max number of boxes to include.
        /// </param>
        /// <param name="iouThreshold">
        /// Minimum value of confidence (0-1) the model must have output for the bounding box to be included.
        /// </param>
        IEnumerable<BoundingBox> FilterBoundingBoxes(IReadOnlyCollection<BoundingBox> boxes, int limit, float iouThreshold);
    }
}