using System;
using System.Collections.Generic;
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

        public NNTrainer()
        {
            m_featureVectorsCache = new List<double[]>();
            m_labelVectorsCache = new List<double[]>();
        }

        public void setFeaturesAndLabels(IEnumerable<double[]> features, IEnumerable<double[]> labels)
        {
            m_features = features;
            m_labels = labels;

            m_featuresEnumerator = m_features.GetEnumerator();
            m_labelsEnumerator = m_labels.GetEnumerator();
        }

        public void initNet(int firstLayerNeurons, int secondLayerNeurons, int outputs)
        {
            //m_featuresEnumerator.Reset();
            //m_labelsEnumerator.Reset();
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

        public void writeModelToFile(String fileURI)
        {
            m_network.Save(fileURI);
        }
    }
}
