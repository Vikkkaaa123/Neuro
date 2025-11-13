namespace MO_32_2_Pereuznik_esc.NeuroNet
{
    //15-71-30-10
    class Network
    {
        //все слои сети
        private InputLayer input_layer=null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(71, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(30, 71, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 30, NeuronType.Output, nameof(output_layer));

        private double[] fact = new double[10]; //массив фактического выхода сети
        private double[] e_error_avr; // среднее значение энергии ошибки эпохи обучения

        //свойства
        public double[] Fact { get => fact; }  //массив фактического выхода сети

        //среднее значение энергии ошибки эпохи обучения
        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }

        //конструкор
        public Network() { }

        //прямой проход сети
        public void ForwardPass(Network net, double[] netInput)
        {
            net.hidden_layer1.Data = netInput;
            net.hidden_layer1.Recognize(null, net.hidden_layer2);
            net.hidden_layer2.Recognize(null, net.output_layer);
            net.output_layer.Recognize(net, null);
        }

        //непосредственно обучение
        public void Train(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Train); //инициализация входного слоя для формирования
            int epoches = 20; //количество эпох обучения
            double tmpSumError; //временная переменная суммы ошибок
            double[] errors; //вектор (массив) сигнала ошибки выходного слоя
            double[] temp_gsums1; //вектор градиента 1-го скрытого слоя
            double[] temp_gsums2; //вектор градиента 2-го скрытого слоя

            e_error_avr = new double[epoches];
            for (int k = 0; k < epoches; k++) //перебор эпох обучения
            {
                e_error_avr[k] = 0; //в начале каждой эпохи обучения значение средней энергии ошибки эпохи обнуляется
                net.input_layer.Shuffling_Array_Rows(net.input_layer.Trainset); //перетасовка обучающейся выборки..
                for (int i = 0; i < net.input_layer.Trainset.GetLength(0); i++)
                {
                    double[] tmpTrain = new double[15]; //обучающий образ
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1];

                    //прямой проход обучающегося выбора
                    ForwardPass(net, tmpTrain);

                    //вычисление ошибки по итерации
                    tmpSumError = 0; //для каждого обучающего образа среднее значение ошибки этого образа обнуляется
                    errors = new double[net.fact.Length]; //переопределение массива сигнала ошибки входного слоя
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == net.input_layer.Trainset[i, 0]) //если номер выходного нейрона совпадает с желаемым откликом
                            errors[x] = 1.0 - net.fact[x];
                        else
                            errors[x] = -net.fact[x]; //errors[x] = 0.0-net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    e_error_avr[k] += tmpSumError / errors.Length; //суммарное значение энергии ошибки k-ой эпохи

                    //Обратный проход и коррекция весов
                    temp_gsums2 = net.output_layer.BackwardPass(errors);
                    temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k]/=net.input_layer.Trainset.GetLength(0); //среднее значение энергии ошибки одной эпохи
            }

            net.input_layer = null; //обнуление (уборка) входного слоя

            //запись скорректированных весов в "память"
            net.hidden_layer1.WeightsInitialize(MemoryMode.SET, nameof(hidden_layer1) + "_memory.csv");
            net.hidden_layer2.WeightsInitialize(MemoryMode.SET, nameof(hidden_layer2) + "_memory.csv");
            net.output_layer.WeightsInitialize(MemoryMode.SET, nameof(output_layer) + "_memory.csv");
        }
    }
}
