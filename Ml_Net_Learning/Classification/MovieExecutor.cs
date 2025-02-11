using Microsoft.ML;
using Microsoft.ML.Trainers;
using System.IO;
using Ml_Net_Learning.Helpers;

namespace Ml_Net_Learning.Classification;

public class MovieExecutor
{
    private readonly string _csvTestFilePath = DirectoryHelper.GetFilePath("test.csv","Csv");
    private readonly string _csvTrainFilePath = DirectoryHelper.GetFilePath("train.csv","Csv");
    private readonly string _zipFolderPath = DirectoryHelper.GetFilePath("","Zip");
    private readonly string _zipFilePath = DirectoryHelper.GetFilePath("sentiment_model.zip","Zip");

    public void Run()
    {
        RunTest();
    }
    
    private void RunTrain()
    {
        var mlContext = new MLContext();

        // string text = File.ReadAllText(_csvFilePath);
        // using (StreamReader sr = new StreamReader(_csvFilePath))
        // {
        //     text = text.Replace("\'", "");
        // }
        // File.WriteAllText(_csvFilePath, text);

        var data = mlContext.Data.LoadFromTextFile<MovieReview>(_csvTrainFilePath, hasHeader: true, allowQuoting: true, separatorChar: ',');

        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text")
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression("Label", "Features"));

        var model = pipeline.Fit(data);
        var predictions = model.Transform(data);
        var metric = mlContext.BinaryClassification.Evaluate(predictions, "Label");
        Console.WriteLine($"Accuracy: {metric.Accuracy}");
        Console.WriteLine($"Precision: {metric.PositivePrecision}");
        Console.WriteLine($"Recall: {metric.PositiveRecall}");
        Console.WriteLine($"F1Score: {metric.F1Score}");

        mlContext.Model.Save(model, data.Schema, _zipFolderPath +"\\sentiment_model.zip");
    }

    private void RunTest()
    {
        var mlContext = new MLContext();

        ITransformer model;
        using (var stream = new FileStream(_zipFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            model = mlContext.Model.Load(stream, out var schema);
        }
        var testData = mlContext.Data.LoadFromTextFile<TextData>(_csvTestFilePath, hasHeader: true, separatorChar: ',');
        var predictor = mlContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);
        var testDataList = mlContext.Data.CreateEnumerable<TextData>(testData, reuseRowObject: false).ToList();

        foreach (var data in testDataList)
        {
            var prediction = predictor.Predict(data);
            Console.WriteLine($"Text: {data.Text} | Prediction: {prediction.SentimentScore} | IsPositiveSentiment: {prediction.IsPositiveSentiment}");
        }
    }
}