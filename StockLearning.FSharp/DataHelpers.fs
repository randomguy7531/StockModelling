namespace StockLearning.FSharp

module DataHelpers =
    open DataPoints
    open System.IO

    let getFileLines fileName =
        let stream = new StreamReader(path = fileName)
        seq
            {
            while not stream.EndOfStream do
            yield stream.ReadLine()
            }

    let readTickerFile fileName =
        let tickerFileContents = getFileLines fileName
        let tickerLine = tickerFileContents |> Seq.head
        let splitters = [' '] |> Seq.toArray
        tickerLine.Split(splitters)  |> Seq.filter(fun x -> not(System.String.IsNullOrWhiteSpace(x))) |> Seq.toList

    let readStockDataFileToCurve fileDir ticker =
        getFileLines(fileDir + ticker + ".csv") 
        |> Seq.map(fun element -> StockPoint.FromCommaDelimitedString(ticker + "," + element)) 
        |> Seq.filter(fun element -> element.IsSome) 
        |> Seq.map(fun element -> element.Value)
        |> Seq.toList 
        |> StockCurve.FromStockPoints

    let readStockDataFilesFromTickers filesDir tickers = 
        let result = tickers |> Seq.map(fun element -> readStockDataFileToCurve filesDir element) |> Seq.toList
        result

    let buildLearningData stockCurves = 
        {stockData = stockCurves}

    let getPosteriorPoints curve dt = 
        curve.Points |> Seq.filter(fun point -> point.Datetime > dt)

    let getPriorPoints curve dt = 
        curve.Points |> Seq.filter(fun point -> point.Datetime < dt)

    let getPointInDateRange curve dtStart dtEnd =
        curve.Points |> Seq.filter(fun point -> point.Datetime > dtStart && point.Datetime < dtEnd)


        