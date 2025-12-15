using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MO_32_2_Pereuznik_esc.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_Layer) :
            base(non, nopn, nt, nm_Layer) { }


        private Random random = new Random();
        public override void dropOut()
        {
            int neuronsToDisable = (int)(numofneurons * 0.5);

            // Просто выбираем случайные нейроны (могут повторяться, но это нормально)
            for (int n = 0; n < neuronsToDisable; n++)
            {
                int randomNeuron = random.Next(numofneurons);

                // Обнуляем все веса этого нейрона
                for (int w = 0; w < neurons[randomNeuron].Weights.Length; w++)
                {
                    neurons[randomNeuron].Weights[w] = 0;
                }
            }
        }


        // прямой проход
        public override void Recognize(Network net, Layer nextLayer)
        {
            double[] hidden_out = new double[numofneurons];
            for (int i = 0; i < numofneurons; i++)
            {
                hidden_out[i] = neurons[i].Output; // Берем текущий выход (может быть 0 если dropOut обнулил)
            }
            nextLayer.Data = hidden_out; //передача выходного сигнала на вход следующего
        }


        // обратный проход
        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];

            // вычисление градиентных сумм  j-го нейрона
            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                {
                    sum += neurons[k].Weights[j + 1] * neurons[k].Derivative * gr_sums[k]; // через градиентные суммы и производную
                }
                gr_sum[j] = sum;
            }

            // цикл коррекции синаптических весов
            for (int i = 0; i < numofneurons; i++)
            {
                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double deltaw;
                    if (n == 0) //если порог
                    {
                        deltaw = momentum * lastdeltaweights[i, 0] + learningrate * neurons[i].Derivative * gr_sums[i];
                    }
                    else
                    {
                        deltaw = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Inputs[n - 1] * neurons[i].Derivative * gr_sums[i];
                    }
                    lastdeltaweights[i, n] = deltaw;
                    neurons[i].Weights[n] += deltaw; // коррекция весов
                }
            }

            return gr_sum;
        }
    }
}