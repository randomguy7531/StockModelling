namespace StockLearning.FSharp

module TrainingData =
    open System
    open DataPoints
    open FeatureBuilders

    //a container to hold the inputs into the model
    type ModelTrainingPoint = 
        {
            symbol : string
            dt : DateTime
            trainingFeatures : double[]
            label : double[]
        }

    //grabs all stock points within a time range to serve as training points
    let getTrainingPoints (dataIn : StockLearningData) dtStart dtEnd = 
        dataIn.stockData 
        |> Seq.collect(fun curve -> DataHelpers.getPointInDateRange curve dtStart dtEnd)

    //for the given overall dataset and stock point, compute the array of doubles to be used as features in the model
    let getFeatureData (dataIn : StockLearningData) (predictionPoint : StockPoint) =
        let currentPointBuilder = FeatureBuilders.CurrentPointMetricFeature dataIn predictionPoint
        let immediatelyPriorBuilder = FeatureBuilders.ImmediatelyPriorPointMetricFeature dataIn predictionPoint

        [
            currentPointBuilder(PointMetrics.Close); 
            currentPointBuilder(PointMetrics.High); 
            currentPointBuilder(PointMetrics.Low); 
            currentPointBuilder(PointMetrics.Open); 
            currentPointBuilder(PointMetrics.Volume);
            immediatelyPriorBuilder(PointMetrics.Close); 
            immediatelyPriorBuilder(PointMetrics.High); 
            immediatelyPriorBuilder(PointMetrics.Low); 
            immediatelyPriorBuilder(PointMetrics.Open); 
            immediatelyPriorBuilder(PointMetrics.Volume);
        ] |> Seq.toArray
        
    //for the given overall dataset and stock point, compute the label for the point (i.e. the thing we are trying to predict)
    let getLabel (dataIn : StockLearningData) (predictionPoint : StockPoint) = 
        let immediatelyPostPoint = 
            dataIn.stockData 
            |> Seq.filter(fun curve -> curve.Symbol = predictionPoint.Symbol)
            |> Seq.collect(fun curve -> DataHelpers.getPosteriorPoints curve predictionPoint.Datetime)
            |> Seq.maxBy(fun point -> predictionPoint.Datetime - point.Datetime)
        //simple case here - we are predicting the probability the stock will close higher than its open tomorrow
        if(immediatelyPostPoint.Close > immediatelyPostPoint.Open) then
            1.0 |> Seq.singleton |> Seq.toArray
        else
            0.0 |> Seq.singleton |> Seq.toArray

    //given a sequence of points, and the overall data, compute the sequence to train the model with
    let buildModelTrainingData (predictionPoints:seq<StockPoint>) (dataIn : StockLearningData) =
        let modelTrainingPoints = 
            predictionPoints
            |> Seq.map(fun point -> {symbol = point.Symbol; dt = point.Datetime; trainingFeatures = getFeatureData dataIn point; label = getLabel dataIn point})
        modelTrainingPoints

    let extractModelInputArray (inputTrainingPoints:seq<ModelTrainingPoint>) =
        inputTrainingPoints
        |> Seq.map(fun point -> point.trainingFeatures)
        |> Seq.toArray

    let extractModelLabelsArray (inputTrainingPoints:seq<ModelTrainingPoint>) =
        inputTrainingPoints
        |> Seq.map(fun point -> point.label)
        |> Seq.toArray