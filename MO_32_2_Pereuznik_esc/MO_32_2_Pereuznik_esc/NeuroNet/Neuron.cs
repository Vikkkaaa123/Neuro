using static System.Math;

//утекающий ReLu

namespace MO_32_2_Pereuznik_esc.NeuroNet
{
    class Neuron
    {
        //поля
        private NeuronType type; //тип нейрона
        private double[] weights; //его веса
        private double[] inputs; //его входы
        private double output; //его выход
        private double derivative; //производная

        //константы для функции активации
        private double a = 0.01d;

        //свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }

        //конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }

        //метод активации нейрона (нелинейное преобразование входного сигнала)
        public void Activator(double[] i)
        {
            inputs = i; // передача вектора входного сигнала в массив входных данных

            double sum = weights[0]; // аффиное преобразование через смещение(нулевой вес - порог)

            for (int j = 0; j < inputs.Length; j++) // цикл вычисления индуцирования
            {
                sum += inputs[j] * weights[j + 1]; // линейные преобразования входных данных
            }


            switch (type)
            {
                case NeuronType.Hidden:         // для нейронов скрытого слоя
                    output = LeakyReLU(sum);
                    derivative = LeakyReLU_Derivativator(sum);
                    break;

                case NeuronType.Output:         // для нейронов выходного слоя
                    output = Exp(sum);          //sum;  Exp(sum); 
                    break;
            }
        }

        //функция активации нейрона
        private double LeakyReLU(double sum)
        {
            output = (sum > 0) ? sum : a * sum;
            return output;
           
        }

        private double LeakyReLU_Derivativator(double sum)
        {
            derivative = (sum > 0) ? 1.0 : a;
            return derivative;
        }

    }
}

