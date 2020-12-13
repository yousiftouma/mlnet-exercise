﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using OnnxObjectDetection.ML.DataModels;

namespace OnnxObjectDetection.Cheating
{
    public class OnnxOutputParserCheat : IOnnxOutputParser
    {
        private class BoundingBoxPrediction : BoundingBoxDimensions
        {
            public float Confidence { get; set; }
        }

        // The number of rows and columns in the grid the image is divided into.
        private const int RowCount = 13;
        private const int ColumnCount = 13;

        // The number of features contained within a box (x, y, height, width, confidence).
        private const int FeaturesPerBox = 5;

        // Labels corresponding to the classes the onnx model can predict. For example, the 
        // Tiny YOLOv2 model included with this sample is trained to predict 20 different classes.
        private readonly string[] _classLabels;

        // Predetermined anchor offsets for the bounding boxes in a cell.
        private readonly (float x, float y)[] _boxAnchors;


        public OnnxOutputParserCheat(IOnnxModel onnxModel)
        {
            _classLabels = onnxModel.Labels;
            _boxAnchors = onnxModel.Anchors;
        }

        // Applies the sigmoid function that outputs a number between 0 and 1.
        private static float Sigmoid(float value)
        {
            var k = MathF.Exp(value);
            return k / (1.0f + k);
        }

        // Normalizes an input vector into a probability distribution.
        private static float[] SoftMax(float[] classProbabilities)
        {
            var max = classProbabilities.Max();
            var exp = classProbabilities.Select(v => MathF.Exp(v - max)).ToList();
            var sum = exp.Sum();
            return exp.Select(v => v / sum).ToArray();
        }

        // Onnx outputs a tensor that has a shape of (for Tiny YOLOv2) 125x13x13. ML.NET flattens
        // this multi-dimensional into a one-dimensional array. This method allows us to access a 
        // specific channel for a given (x,y) cell position by calculating the offset into the array.
        private static int GetOffset(int row, int column, int channel)
        {
            const int channelStride = RowCount * ColumnCount;
            return (channel * channelStride) + (column * ColumnCount) + row;
        }

        // Extracts the bounding box features (x, y, height, width, confidence) method from the model
        // output. The confidence value states how sure the model is that it has detected an object. 
        // We use the Sigmoid function to turn that confidence into a percentage.
        private static BoundingBoxPrediction ExtractBoundingBoxPrediction(float[] modelOutput, int row, int column, int channel)
        {
            return new BoundingBoxPrediction
            {
                X = modelOutput[GetOffset(row, column, channel++)],
                Y = modelOutput[GetOffset(row, column, channel++)],
                Width = modelOutput[GetOffset(row, column, channel++)],
                Height = modelOutput[GetOffset(row, column, channel++)],
                Confidence = Sigmoid(modelOutput[GetOffset(row, column, channel++)])
            };
        }

        // The predicted x and y coordinates are relative to the location of the grid cell; we use 
        // the logistic sigmoid to constrain these coordinates to the range 0 - 1. Then we add the
        // cell coordinates (0-12) and multiply by the number of pixels per grid cell (32).
        // Now x/y represent the center of the bounding box in the original 416x416 image space.
        // Additionally, the size (width, height) of the bounding box is predicted relative to the
        // size of an "anchor" box. So we transform the width/weight into the original 416x416 image space.
        private BoundingBoxDimensions MapBoundingBoxToCell(int row, int column, int box, BoundingBoxPrediction boxDimensions)
        {
            const float cellWidth = ImageSettings.imageWidth / ColumnCount;
            const float cellHeight = ImageSettings.imageHeight / RowCount;

            var mappedBox = new BoundingBoxDimensions
            {
                X = (row + Sigmoid(boxDimensions.X)) * cellWidth,
                Y = (column + Sigmoid(boxDimensions.Y)) * cellHeight,
                Width = MathF.Exp(boxDimensions.Width) * cellWidth * _boxAnchors[box].x,
                Height = MathF.Exp(boxDimensions.Height) * cellHeight * _boxAnchors[box].y,
            };

            // The x,y coordinates from the (mapped) bounding box prediction represent the center
            // of the bounding box. We adjust them here to represent the top left corner.
            mappedBox.X -= mappedBox.Width / 2;
            mappedBox.Y -= mappedBox.Height / 2;

            return mappedBox;
        }

        // Extracts the class predictions for the bounding box from the model output using the
        // GetOffset method and turns them into a probability distribution using the SoftMax method.
        private float[] ExtractClassProbabilities(float[] modelOutput, int row, int column, int channel, float confidence)
        {
            var classProbabilitiesOffset = channel + FeaturesPerBox;
            var classProbabilities = new float[_classLabels.Length];

            for (var classProbability = 0; classProbability < _classLabels.Length; classProbability++)
            {
                classProbabilities[classProbability] = modelOutput[GetOffset(row, column, classProbability + classProbabilitiesOffset)];
            }

            return SoftMax(classProbabilities).Select(p => p * confidence).ToArray();
        }

        // IoU (Intersection over union) measures the overlap between 2 boundaries. We use that to
        // measure how much our predicted boundary overlaps with the ground truth (the real object
        // boundary). In some datasets, we predefine an IoU threshold (say 0.5) in classifying
        // whether the prediction is a true positive or a false positive. This method filters
        // overlapping bounding boxes with lower probabilities.
        private static float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;
            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaA <= 0 || areaB <= 0)
            {
                return 0;
            }

            var minX = MathF.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = MathF.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = MathF.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = MathF.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = MathF.Max(maxY - minY, 0) * MathF.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        public IReadOnlyCollection<BoundingBox> ParseOutputs(float[] modelOutput, float probabilityThreshold = .3f)
        {
            var boxes = new List<BoundingBox>();

            for (var row = 0; row < RowCount; row++)
            {
                for (var column = 0; column < ColumnCount; column++)
                {
                    for (var box = 0; box < _boxAnchors.Length; box++)
                    {
                        var channel = box * (_classLabels.Length + FeaturesPerBox);

                        var boundingBoxPrediction = ExtractBoundingBoxPrediction(modelOutput, row, column, channel);

                        var mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxPrediction);

                        if (boundingBoxPrediction.Confidence < probabilityThreshold)
                        {
                            continue;
                        }

                        var classProbabilities = ExtractClassProbabilities(modelOutput, row, column, channel, boundingBoxPrediction.Confidence);

                        var (topProbability, topIndex) = classProbabilities.Select((probability, index) => (Score: probability, Index: index)).Max();

                        if (topProbability < probabilityThreshold)
                        {
                            continue;
                        }

                        boxes.Add(new BoundingBox
                        {
                            Dimensions = mappedBoundingBox,
                            Confidence = topProbability,
                            Label = _classLabels[topIndex],
                            BoxColor = BoundingBox.GetColor(topIndex)
                        });
                    }
                }
            }
            return boxes;
        }

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
        public IEnumerable<BoundingBox> FilterBoundingBoxes(IReadOnlyCollection<BoundingBox> boxes, int limit, float iouThreshold)
        {
            var results = new List<BoundingBox>();
            var filteredBoxes = new bool[boxes.Count];
            var sortedBoxes = boxes.OrderByDescending(b => b.Confidence).ToArray();

            for (var i = 0; i < boxes.Count; i++)
            {
                if (filteredBoxes[i])
                {
                    continue;
                }

                results.Add(sortedBoxes[i]);

                if (results.Count >= limit)
                {
                    break;
                }

                for (var j = i + 1; j < boxes.Count; j++)
                {
                    if (filteredBoxes[j])
                    {
                        continue;
                    }

                    if (IntersectionOverUnion(sortedBoxes[i].Rect, sortedBoxes[j].Rect) > iouThreshold)
                    {
                        filteredBoxes[j] = true;
                    }

                    if (filteredBoxes.Count(b => b) <= 0)
                    {
                        break;
                    }
                }
            }
            return results;
        }
    }
}
