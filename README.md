# ML.NET exercise

Based on [this](https://github.com/dotnet/machinelearning-samples/releases/tag/186179).

## The exercise

This is an exercise in ML.NET and ONNX models.
The goal of the exercise is to get a basic understanding in how the ML.NET framework works and how to integrate a neural network in a .NET core project.

In this sample, which is based on the end-to-end C# samples over at [the ML.NET samples](https://github.com/dotnet/machinelearning-samples), a neural network is used to detect objects in images through an ASP.NET webapp where you can choose from a couple of preloaded images, or upload an image of your choosing.

**Note that the main branch contains a working solution, refer to the steps below to get the full experience. Also, avoid the diffs in commit history ;).**

## Prerequisites

* .NET Core 3.1 SDK which can be downloaded [here](https://dotnet.microsoft.com/download/dotnet-core/3.1).

* An IDE or text editor of your choice (a .sln file is included for Visual Studio).

* The requirements listed [here](https://github.com/microsoft/onnxruntime/tree/v1.4.0#system-requirements) (for CPU to begin with for this exercise).

* Fork and/or clone the repo.

Try running the project by either

* Visual Studio
  
  Open the solution, run the default configuration for _OnnxObjectDetectionWeb_.
  
  A browser should open with the app, otherwise just go to http://localhost:5000 in your browser of choice.

* dotnet cli
  
  Navigate to src/OnnxObjectDetectionWeb and run _dotnet run_.

  Go to http://localhost:5000 in your browser of choice.

Try it out a bit and see what it does.

Hopefully, you now have an idea of the goal!

## Part 1

Checkout the **part1** branch.

If you try running the project, you will notice that nothing works right now.

We have a lot of stuff to do to get it up and running, and hopefully, by doing so, we will learn a great deal about ML.NET!

To help out with some tedious parts that are inherited given the actual ONNX model being used in this sample, there are a bunch of helpers under the _OnnxObjectDetection.Cheating_ namespace.

Actually, most TODOs can be solved using these, but the ones where I recommend using it are specifically noted as such.

### Task 1 - Data models

To get all data modelling correct, we need to know how our model works. 
You can find more information about the model that is used in this project [here](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2).

Another great thing to do when working with ONNX models is to inspect the actual model.
This can be done using for example [this app](https://netron.app).
Load the ONNX model (OnnxObjectDetection/ML/OnnxModels/TinyYolo2_model.onnx) and inspect it a bit.

The parts that are important to note are the names are of the inputs and outputs, as well as the form of what is expected as input to the model and the form of the output.

### Task 2 - your first TODOs

Let's use this new information we learned from the model and apply it where we have the first TODOs.

Go to OnnxObjectDetection/ML/DataModels and fix the TODOs in _TinyYoloModel_ and _ImageInputData_.

Use [this](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.image.imagetypeattribute?view=ml-dotnet) to annotate the Bitmap properly.

ML.NET supports Bitmaps for images and needs further information about the shape of the actual image for transforms later on.

### Task 3 - Creating a transformation pipeline model

Okay, so we got the models ready to go, time to define a pipeline that takes an image and does the magic.

Go to OnnxObjectDetection/ML/OnnxModelConfigurator.

From task 1, we got the information with us that the ONNX model expected input on the form 3x416x416 of floats.

This means we somehow need to make sure the image is of the correct size, as well as transformed to a 3d array of floats before being fed to the actual model.

This is where transforms come in!
An MLContext object has a Transforms property which contains a bunch of transformers (or estimators).
Transformers are basically pure functions that are applied, transforming the input into an output.
These can then be appended to each other, creating a transformation pipeline.

Now I won't lie, there is a bunch of behind the scenes-stuff I haven't delved into with these, but learning what transforms are available and their APIs will take you a long way!

I've prepared methods returning the correct type of the estimators you need, it's now your job to figure out how to implement them and the order we should use them in.

### Task 3a

Implement the 4 transformers and append them to the pipeline in the correct order.
The following links will help figuring out the APIs.

[here](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.imageestimatorscatalog?view=ml-dotnet)
[here](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.onnxcatalog.applyonnxmodel?view=ml-dotnet)
and
[here](https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.columncopyingestimator?view=ml-dotnet)

Remember that there was a name on the output from the OnnxModel?

When the pipeline finishes, the last output column is expected to map to the name of a property where the output data will be stored.

The method _GetMlNetPredictionEngine_ gives a hint of what will be input and output of the full pipeline.
Use this information to figure out the name of the final output column of the pipeline.

### Task 3b

Implement the public method to get a prediction engine with the full transformer pipeline.

To do this we once again need to use the MLContext object.
It also has a Model property with some convenient methods.
Check them out to solve this.

## Does it work?

Try it out! By now you should be able to classify the preloaded images and upload your own images.

Note that uploading your own images seems to mess upp the scaling a bit when it comes to labels and boxes, so you might have to zoom in on the image to see them.

## Intermission

You can now choose to either checkout the **part2** branch, which will probably differ a bit from how you ended up after finishing part 1, or you can choose to keep working from your branch which may make the next part a bit trickier, depending on how well your solution is aligned with the tasks in part 2.

## Part 2

So it works! Problem is, we're loading the model on each new request.
If you check memory usage, you'll see that the program is eating memory and making the GC sweat.
Each request also takes a bit of time.
Let's see if we can improve it!

These tasks are based on a working solution to part 1 which can be found on branch **part2**, including a set of TODOs.

Feel free to accomplish the tasks differently if you chose to keep working from your own solution!
Hopefully, it is easy to understand what to do without the TODOs.

The goal is to stop creating a new pipeline model each request and instead utilize the _PredictionEnginePool_ and Dependency Injection to both speed things up and be more efficient with our resources.

### Task 1

Go to _Startup_ in the web project and notice there are a couple of TODOs.
The first one concerns creating a pipeline and storing it, while the second one concerns registering something in the service collection to be used with Dependency Injection.

Make sure the pipeline model is now created here and not in _ObjectDetectionService_.

### Task 2

To be able to save the model, we need to implement the corresponding method in _OnnxModelConfigurator_.

Go ahead and implement the method.
Look [here](https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/save-load-machine-learning-models-ml-net) for help.

### Task 3

Save the model somewhere (perhaps next to the onnx file?).

### Task 4

Register a PredictionEnginePool in the service collection so we can inject it in _ObjectDetectionService_. Remember where you saved the zip? You'll need to use the extension method that loads a model from a (zip) file.

Look [here](https://docs.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/serve-model-web-api-ml-net#register-predictionenginepool-for-use-in-the-application) for help.

### Task 5

Go back to _ObjectDetectionService_ and exchange the manually created prediction engine for a prediction engine pool that is injected in the constructor of the class.

The pool has the same interface for getting predictions as the "normal" prediction engine.
