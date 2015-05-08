using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace StockLearning.ModelTraining
{
    public class NNTrainer
    {
        private IEnumerable<double[]> m_features;
        private IEnumerable<double[]> m_labels;

        private IEnumerator<double[]> m_featuresEnumerator;
        private IEnumerator<double[]> m_labelsEnumerator;

        private List<double[]> m_featureVectorsCache;
        private List<double[]> m_labelVectorsCache;

        private ActivationNetwork m_network;
        private BackPropagationLearning m_teacher;

        /// <summary>
        /// Constructor
        /// </summary>
        public NNTrainer()
        {
            m_featureVectorsCache = new List<double[]>();
            m_labelVectorsCache = new List<double[]>();
        }

        /// <summary>
        /// sets the enumerables used to train the neural net (these enumerables are cached in Lists after calling RunEpoch())
        /// </summary>
        /// <param name="features">enumerable of the feature/input vectors to be used for each training sample</param>
        /// <param name="labels">enumerable of the outputs to be used as labels for each training sample. Length must match that of features param.</param>
        public void setTrainingFeaturesAndLabels(IEnumerable<double[]> features, IEnumerable<double[]> labels)
        {
            m_features = features;
            m_labels = labels;

            m_featuresEnumerator = m_features.GetEnumerator();
            m_labelsEnumerator = m_labels.GetEnumerator();
        }

        /// <summary>
        /// create the necessary neural net and trainer objects prior to training
        /// </summary>
        /// <param name="firstLayerNeurons">Number of neurons to use in the first hidden layer</param>
        /// <param name="secondLayerNeurons">Number of neurons to use in the second hidden layer</param>
        /// <param name="outputs">number of outputs/labels expected out of this net</param>
        public void initNet(int firstLayerNeurons, int secondLayerNeurons, int outputs)
        {
            m_featuresEnumerator.MoveNext();
            m_labelsEnumerator.MoveNext();

            m_featureVectorsCache.Add(m_featuresEnumerator.Current);
            m_labelVectorsCache.Add(m_labelsEnumerator.Current);

            if (m_featureVectorsCache.Count() == 0)
            {
                throw new InvalidOperationException("Cannot init net while feature vector is not initialized!");
            }

            int inputLayerNeurons = m_featureVectorsCache[0].Length;

            if (secondLayerNeurons > 0)
            {
                m_network = new ActivationNetwork(new SigmoidFunction(), inputLayerNeurons, firstLayerNeurons, secondLayerNeurons, outputs);
            }
            else
            {
                m_network = new ActivationNetwork(new SigmoidFunction(), inputLayerNeurons, firstLayerNeurons, outputs);
            }

            m_teacher = new BackPropagationLearning(m_network);
        }

        /// <summary>
        /// Runs an iteration of training on the neural net using the previously set feature and label enums. 
        /// Note these enums are cached internally to this class after the first call to this function, and are not enumerated again.
        /// </summary>
        /// <returns>Double representing the sum squared error of the epoch divided by the number of training samples</returns>
        public double runEpoch()
        {
            double errorSum = 0;

            //check for unequal length feature/label lists before proceeding
            if (m_featureVectorsCache.Count() != m_labelVectorsCache.Count())
            {
                throw new InvalidOperationException("Cannot train model with unequal length feature and label lists!");
            }

            //first, run through the cached data
            for (int i = 0; i < m_featureVectorsCache.Count() && i < m_labelVectorsCache.Count(); i++)
            {
                errorSum += m_teacher.Run(m_featureVectorsCache[i], m_labelVectorsCache[i]);
            }

            //then, enumate the enumerator (if we havent already) and cache into the lists, while training the net
            bool featuresMoved = m_featuresEnumerator.MoveNext();
            bool labelsMoved = m_labelsEnumerator.MoveNext();
            while(featuresMoved && labelsMoved)
            {
                double[] elementFeatures = m_featuresEnumerator.Current;
                double[] elementLabels = m_labelsEnumerator.Current;

                errorSum += m_teacher.Run(elementFeatures, elementLabels);

                m_featureVectorsCache.Add(elementFeatures);
                m_labelVectorsCache.Add(elementLabels);

                featuresMoved = m_featuresEnumerator.MoveNext();
                labelsMoved = m_labelsEnumerator.MoveNext();

                if(featuresMoved != labelsMoved)
                {
                    throw new InvalidOperationException("Cannot train model with unequal length feature and label lists!");
                }
            }


            return errorSum / m_featureVectorsCache.Count();
        }

        /// <summary>
        /// For the training data cached in this class, compute the outputs generated by the neural net in its current state. 
        /// Note that if the training data is not cached yet via calling RunEpoch(), this function will return an empty list.
        /// </summary>
        /// <returns>A List containing one double[][] for each training sample. Each list member should have 3 subs arrays. 
        /// The array at [0] contains the features used as inputs to the net.
        /// The array at [1] contains the results obtained at the output of the neural net using the features in [0].
        /// The array at [2] contains the labels that were expected to be seen at the output.</returns>
        public List<double[][]> GetTrainingOutputs()
        {
            List<double[][]> toReturn = new List<double[][]>();

            for(int i = 0; i < m_featureVectorsCache.Count(); i++)
            {
                double[] features = m_featureVectorsCache[i];
                double[] results = m_network.Compute(features);
                double[] labels = m_labelVectorsCache[i];

                double[][] toAdd = { features, results, labels };

                toReturn.Add(toAdd);
            }

            return toReturn;
        }
        /// <summary>
        /// For an arbitrary set of features and labels, compute the outputs generated by the neural net in its current state.
        /// </summary>
        /// <param name="validationFeatures">enumerable of the feature/input vectors to be used for each validation sample<</param>
        /// <param name="validationLabels">enumerable of the outputs to be used as labels for each validation sample. Length must match that of features param.</param>
        /// <returns>A List containing one double[][] for each training sample. Each list member should have 3 subs arrays. 
        /// The array at [0] contains the features used as inputs to the net.
        /// The array at [1] contains the results obtained at the output of the neural net using the features in [0].
        /// The array at [2] contains the labels that were expected to be seen at the output.</returns>
        public List<double[][]> GetValidationOutputs(IEnumerable<double[]> validationFeatures, IEnumerable<double[]> validationLabels)
        {
            List<double[][]> toReturn = new List<double[][]>();

            IEnumerator<double[]> validationFeatureEnumerator = validationFeatures.GetEnumerator();
            IEnumerator<double[]> validationLabelEnumerator = validationLabels.GetEnumerator();

            bool featuresMoved = validationFeatureEnumerator.MoveNext();
            bool labelsMoved = validationLabelEnumerator.MoveNext();

            while (featuresMoved && labelsMoved)
            {
                double[] featureVector = validationFeatureEnumerator.Current;
                double[] resultVector = m_network.Compute(featureVector);
                double[] labelVector = validationLabelEnumerator.Current;
                double[][] toAdd = { featureVector, resultVector, labelVector };

                toReturn.Add(toAdd);

                featuresMoved = validationFeatureEnumerator.MoveNext();
                labelsMoved = validationLabelEnumerator.MoveNext();

                if (featuresMoved != labelsMoved)
                {
                    throw new InvalidOperationException("Cannot validate model on unequal length feature and label lists!");
                }
            }

            return toReturn;
        }
        /// <summary>
        /// Writes the resutls from the training or validation sets to the given file location in .csv format.
        /// </summary>
        /// <param name="results">Results to use. Must be a non empty list with each double[][] having a top level array length of 3.</param>
        /// <param name="fileURI">File location to write the data to.</param>
        public void WriteResultsToFile(List<double[][]> results, String fileURI)
        {
            String toWrite = "";

            int numFeatures = -1;
            int numOutputs = -1;
            int numLabels = -1;

            if(results == null || results.Count() == 0)
            {
                throw new InvalidOperationException("Cannot output results from a null or empty results list!");
            }

            for(int i = 0; i < results.Count(); i++)
            {
                double[][] row = results[i];

                double[] features = row[0];
                double[] outputs = row[1];
                double[] labels = row[2];

                //on first run through, determine size of feature, output, and labels arrays, and write header data to file string
                if(numFeatures == -1 && numOutputs == -1 && numLabels == -1)
                {
                    numFeatures = features.Length;

                    for (int j = 0; j < numFeatures; j++ )
                    {
                        toWrite += "Feature_" + j + ",";
                    }

                    numOutputs = outputs.Length;

                    for (int j = 0; j < numOutputs; j++)
                    {
                        toWrite += "Output_" + j + ",";
                    }

                    numLabels = labels.Length;

                    for (int j = 0; j < numLabels; j++)
                    {
                        toWrite += "Label_" + j + ",";
                    }
                    toWrite += "\r\n";
                }

                if(features.Length != numFeatures)
                    throw new InvalidOperationException("Cannot output results if feature vector length is not consistent!");

                if (outputs.Length != numOutputs)
                    throw new InvalidOperationException("Cannot output results if output/result vector length is not consistent!");

                if (labels.Length != numLabels)
                    throw new InvalidOperationException("Cannot output results if labels vector length is not consistent!");


                //write the row output in comma delimited format.
                for (int j = 0; j < numFeatures; j++)
                {
                    toWrite += features[j].ToString() + ",";
                }

                for (int j = 0; j < numOutputs; j++)
                {
                    toWrite += outputs[j].ToString() + ",";
                }

                for (int j = 0; j < numLabels; j++)
                {
                    toWrite += labels[j].ToString() + ",";
                }

                toWrite += "\r\n";
            }

            //write final contents to file
            File.WriteAllText(fileURI, toWrite);
        }
        /// <summary>
        /// Writes the model in its current state to a file.
        /// </summary>
        /// <param name="fileURI">String representation of location to write file to.</param>
        public void writeModelToFile(String fileURI)
        {
            m_network.Save(fileURI);
        }
    }
}
