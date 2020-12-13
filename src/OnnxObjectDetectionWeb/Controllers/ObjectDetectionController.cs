using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using OnnxObjectDetectionWeb.Services;
using OnnxObjectDetection;
using OnnxObjectDetection.ML.DataModels;
using OnnxObjectDetectionWeb.ImageFileHelpers;
using OnnxObjectDetectionWeb.Utilities;

namespace OnnxObjectDetectionWeb.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ObjectDetectionController : ControllerBase
    {
        private readonly string _imagesTmpFolder;        

        private readonly ILogger<ObjectDetectionController> _logger;
        private readonly IObjectDetectionService _objectDetectionService;

        public ObjectDetectionController(IObjectDetectionService objectDetectionService, ILogger<ObjectDetectionController> logger)
        {
            //Get injected dependencies
            _objectDetectionService = objectDetectionService;
            _logger = logger;
            _imagesTmpFolder = CommonHelpers.GetAbsolutePath(@"../../../ImagesTemp");
        }

        public class Result
        {
            public string ImageString { get; set; }
        }

        [HttpGet()]
        public IActionResult Get([FromQuery]string url)
        {
            var imageFileRelativePath = @"../../../assets" + url;
            var imageFilePath = CommonHelpers.GetAbsolutePath(imageFileRelativePath);
            try
            {
                var image = Image.FromFile(imageFilePath);
                //Convert to Bitmap
                var bitmapImage = (Bitmap)image;

                //Set the specific image data into the ImageInputData type used in the DataView
                var imageInputData = new ImageInputData { Image = bitmapImage };

                //Detect the objects in the image                
                var result = DetectAndPaintImage(imageInputData, imageFilePath);
                return Ok(result);
            }
            catch (Exception e)
            {
                _logger.LogInformation("Error is: " + e.Message);
                return BadRequest();
            }
        }

        [HttpPost]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [Route("IdentifyObjects")]
        public async Task<IActionResult> IdentifyObjects(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
            {
                return BadRequest();
            }

            try
            {
                Image image;
                await using (var imageMemoryStream = new MemoryStream())
                {
                    await imageFile.CopyToAsync(imageMemoryStream);                

                    //Check that the image is valid
                    var imageData = imageMemoryStream.ToArray();
                    if (!imageData.IsValidImage())
                    {
                        return StatusCode(StatusCodes.Status415UnsupportedMediaType);
                    }

                    //Convert to Image
                    image = Image.FromStream(imageMemoryStream);
                }

                var fileName = $"{image.GetHashCode()}.Jpeg";
                var imageFilePath = Path.Combine(_imagesTmpFolder, fileName);
                //save image to a path
                image.Save(imageFilePath, ImageFormat.Jpeg);

                //Convert to Bitmap
                var bitmapImage = (Bitmap)image;

                _logger.LogInformation($"Start processing image...");

                //Measure execution time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                //Set the specific image data into the ImageInputData type used in the DataView
                var imageInputData = new ImageInputData { Image = bitmapImage };

                //Detect the objects in the image                
                var result = DetectAndPaintImage(imageInputData, imageFilePath);

                //Stop measuring time
                watch.Stop();
                var elapsedMs = watch.ElapsedMilliseconds;
                _logger.LogInformation($"Image processed in {elapsedMs} milliseconds");
                return Ok(result);
            }
            catch (Exception e)
            {
                _logger.LogInformation("Error is: " + e.Message);
                return BadRequest();
            }
        }

        private Result DetectAndPaintImage(ImageInputData imageInputData, string imageFilePath)
        {
            //Predict the objects in the image
            var boundingBoxes = _objectDetectionService.DetectObjectsUsingModel(imageInputData);
            var img = _objectDetectionService.DrawBoundingBox(imageFilePath, boundingBoxes);

            using var m = new MemoryStream();

            img.Save(m, img.RawFormat);
            var imageBytes = m.ToArray();

            // Convert byte[] to Base64 String
            var base64String = Convert.ToBase64String(imageBytes);
            var result = new Result { ImageString = base64String };
            return result;
        }
    }
}
