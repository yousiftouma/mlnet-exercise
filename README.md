# ML.NET exercise

Based on [this](https://github.com/dotnet/machinelearning-samples/releases/tag/186179).

## The exercise

This is an exercise in ML.NET and ONNX models.
The goal of the exercise is to get a basic understanding in how the ML.NET framework works and how to integrate a neural network in a .NET core project.

In this sample, which is based on the end-to-end C# samples over at [the ML.NET samples](https://github.com/dotnet/machinelearning-samples), a neural network is used to detect objects in images through an ASP.NET webapp where you can choose from a couple of preloaded images, or upload an image of your choosing.

**Note that the main branch contains a working solution, refer to the steps below to get the full experience. Also, avoid the diffs in commit history ;).**

## Prerequisites

.NET Core 3.1 SDK which can be downloaded [here](https://dotnet.microsoft.com/download/dotnet-core/3.1).

An IDE or text editor of your choice (a .sln file is included for Visual Studio).

The requirements listed [here](https://github.com/microsoft/onnxruntime/tree/v1.4.0#system-requirements) (for CPU to begin with for this exercise).

Try running the project by either

* For Visual Studio
  
  Open the solution, run the default configuration for _OnnxObjectDetectionWeb_.

* dotnet cli
  
  Navigate to src/OnnxObjectDetectionWeb and run _dotnet run_.

Hopefully, you now have an idea of the goal!

## Part 1

* Checkout the **part1** branch.

## Part 2

So it works! Problem is, we're loading the model on each new request.
If you check memory usage, you'll see that the program is eating memory and making the GC sweat.
Each request also takes a bit of time.
Let's see if we can improve it!

These tasks are based on a working solution to part 1.
Feel free to accomplish the tasks differently if you chose to keep working from your own solution!

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
