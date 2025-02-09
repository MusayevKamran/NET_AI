using Microsoft.ML;
using Microsoft.ML.Data;
using Ml_Net_Learning.Helpers;

namespace Ml_Net_Learning.Housing;

public class HousingExecutor
{
    private readonly string _csvFilePath = DirectoryHelper.GetProjectRoot("housing-data.csv");

    public void Run()
    {
        var mlContext = new MLContext();

        var data = mlContext.Data.LoadFromTextFile<HousingData>(_csvFilePath, separatorChar: ',', hasHeader: true);

        var dataPipeline = mlContext.Transforms.Conversion.ConvertType("SquareFeet", outputKind: DataKind.Single)
            .Append(mlContext.Transforms.NormalizeMinMax("SquareFeet"))
            .Append(mlContext.Transforms.Concatenate("Features", "SquareFeet", "Bedrooms"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"));

        var transformedData = dataPipeline.Fit(data).Transform(data);
        var transformedDataEnumerable = mlContext.Data.CreateEnumerable<TransformedHousingData>(transformedData, reuseRowObject: false).ToList();
        foreach (var item in transformedDataEnumerable)
        {
            Console.WriteLine($"SquareFeet: {item.SquareFeet}, Bedrooms: {item.Bedrooms}, Price: {item.Price}, Features: [{string.Join(", ", item.Features)}], Neighborhood: [{string.Join(", ", item.Neighborhood)}]");
        }
    }
}