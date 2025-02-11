using Microsoft.ML;
using Microsoft.ML.Data;
using Ml_Net_Learning.Helpers;

namespace Ml_Net_Learning.DataPoint;

public class DataPointExecutor
{
    private readonly string _csvFilePath = DirectoryHelper.GetFilePath("data.csv", "Csv");

    public void Run()
    {
        var mlContext = new MLContext();

        var data = mlContext.Data.LoadFromTextFile<DataPoint>(_csvFilePath, separatorChar: ',', hasHeader: true);

        var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        // Feature and label transformations
        var featurePipeline = mlContext.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean)
            .Append(mlContext.Transforms.Concatenate("Features", nameof(DataPoint.Feature1), nameof(DataPoint.Feature2)));

        var logisticRegressionPipeline = featurePipeline
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", maximumNumberOfIterations: 100));

        var fastTreePipeline = featurePipeline
            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", numberOfLeaves: 20, numberOfTrees: 100));


        Console.WriteLine("Training Logistic Regression Model");
        var logisticRegressionModel = logisticRegressionPipeline.Fit(trainTestSplit.TrainSet);
        Console.WriteLine("Training Fast Tree Model");
        var fastTreeModel = fastTreePipeline.Fit(trainTestSplit.TrainSet);
        Console.WriteLine("Evaluating Logistic Regression Model");
        var logisticRegressionPredictions = logisticRegressionModel.Transform(trainTestSplit.TestSet);
        var logisticRegressionMetrics = mlContext.BinaryClassification.Evaluate(logisticRegressionPredictions);
        Prediction.EvaluateMetrics("Logistic Regression", logisticRegressionMetrics);
        Console.WriteLine("Evaluating Fast Tree Model");
        var fastTreePredictions = fastTreeModel.Transform(trainTestSplit.TestSet);
        var fastTreeMetrics = mlContext.BinaryClassification.Evaluate(fastTreePredictions);
        Prediction.EvaluateMetrics("Fast Tree", fastTreeMetrics);

        if (logisticRegressionMetrics.Accuracy > fastTreeMetrics.Accuracy)
            Console.WriteLine("Logistic Regression Model is better");
        else if (logisticRegressionMetrics.Accuracy < fastTreeMetrics.Accuracy)
            Console.WriteLine("Fast Tree Model is better");
        else
            Console.WriteLine("Logistic Regression and Fast Tree Model is equally good");
    }
}