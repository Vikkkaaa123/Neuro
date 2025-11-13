using System;
using System.IO;
using System.Windows.Forms;

namespace MO_32_2_Pereuznik_esc.NeuroNet
{
    abstract class Layer //модификаторы protected стоят для внутрииерархического использования членов
    {
        //Поля
        protected string name_Layer; //наименование слоя, которое используется для связи с одноименными слоями системы
        string pathDirWeights; // путь к каталогу, где находится файл синаптических весов для загрузки и сохранения
        string pathFileWeights; // путь к файлу синаптических вестов для нейросети
        protected int numofneurons; //число нейронов текущего слоя
        protected int numofprevneurons; //число нейронов предыдущего слоя
        protected const double learningrate = 0.012; //скорость обучения
        protected const double momentum = 0.050d; //момент итерации
        protected double[,] lastdeltaweights; //веса предыдущей итерации обучения
        protected Neuron[] neurons; //массив нейронов текущего слоя

        //Свойства
        public Neuron[] Neurons { get => neurons; set => neurons = value; } //массив нейронов слоя
        public double[] Data //передача входных данных на нейроны слоя и активации нейронов
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                {
                    Neurons[i].Activator(value);
                }
            }
        }

        //конструктор
        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer) //nm_Layer используется для идентификации слоя
        {
            //int i, j; //счетчик циклов
            numofneurons = non; //количество нейронов текущего слоя
            numofprevneurons = nopn; //количество нейронов предыдущего слоя
            Neurons = new Neuron[non]; //определение массива нейронов
            name_Layer = nm_Layer; //наименование слоя, которое используется для связи с одноименными файлами весов
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\"; //путь к каталогу, где хранятся файлы весов 
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv"; //путь к фалу синаптических  весов текущего слоя

            lastdeltaweights = new double[non, nopn + 1];
            double[,] Weights; //временный массив синаптическихх весов текущего слоя

            if (File.Exists(pathDirWeights)) //опрделяет, существует ли pathFileWeights
                Weights = WeightsInitialize(MemoryMode.GET, pathFileWeights); //считывание
            else
            {
                Directory.CreateDirectory(pathDirWeights); //создание директории
                Weights = WeightsInitialize(MemoryMode.INIT, pathFileWeights); //если
            }
            for (int i = 0; i < non; i++) //цикл формирования нейронов слоя и заполнение
            {
                double[] tmp_weights = new double[nopn + 1];
                for (int j = 0; j < nopn; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i]=new Neuron(tmp_weights, nt); //заполнение массивов нейронами
            }
            WeightsInitialize(MemoryMode.SET, pathFileWeights);
        }

        //Метод работы с массивом синаптических весов слоя
        public double[,] WeightsInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' }; //разделитель слоев
            //string tmpStr; //временная строка для чтения
            string[] tmpStrWeights;  //временный массив строк
            Random random = new Random();
            double[,] weights = new double[numofneurons, numofprevneurons + 1]; //массив синаптических весов

            switch (mm)
            {
                case MemoryMode.GET:
                    tmpStrWeights=File.ReadAllLines(path); //считывание строк текстового массива весов
                    string[] memory_elemnt;
                    for (int i = 0; i < numofneurons; i++)
                    {
                        memory_elemnt = tmpStrWeights[i].Split(delim); //разбивает строку на элементы в
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i,j]=double.Parse(memory_elemnt[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture); //преобразование строкии к единому образцу чисел
                        }
                    }
                    break;

                case MemoryMode.SET:
                    string Str = "";
                    
                        for (int i = 0; i < numofneurons; i++)
                        {
                        Str += Neurons[i].Weights[0].ToString();
                            for (int j = 1; j < numofprevneurons + 1; j++)
                            {
                            Str += ";" + Neurons[i].Weights[j].ToString();
                            }
                        Str += "\n";
                        }
                    File.WriteAllText(path, Str);
                    break;

                case MemoryMode.INIT:
                    // инициализация синаптических весов случайными значениями для начала обучения
                    for (int i = 0; i < numofneurons; i++)
                    {
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = random.NextDouble() * 2 - 1; // случайные значения от -1 до 1
                        }
                    }
                    break;
            }
            return weights;
        }

        //метод расчета среднего
        protected double Calc_Average(double[] arr)
        {
            if (arr == null || arr.Length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < arr.Length; i++)
            {
                sum += arr[i];
            }
            return sum / arr.Length;
        }

        //метод расчета дисперсии
        protected double Calc_Dispers(double[] arr)
        {
            if (arr == null || arr.Length == 0)
                return 0;

            double average = Calc_Average(arr);
            double sumSquares = 0;

            for (int i = 0; i < arr.Length; i++)
            {
                double deviation = arr[i] - average;
                sumSquares += deviation * deviation;
            }

            return sumSquares / arr.Length;
        }


        abstract public void Recognize(Network net, Layer nextLayer); //для прямых проходов

        abstract public double[] BackwardPass(double[] stuff); //для обратных
    }
}
