using System;
using System.IO;

namespace MO_32_2_Pereuznik_esc.NeuroNet
{
    class InputLayer
    {
        //поля
        private double[,] trainset; //100 изображений в обучающейся выборке (110)
        private double[,] testset; //10 изображений в тестовой выборке

        //свойства
        public double[,] Trainset { get => trainset; }
        public double[,] Testset { get => testset; }

        //конструктор
        public InputLayer(NetworkMode nm)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory; //директория
            string[] tmpArrStr; //временный массив строк
            string[] tmpStr; //временный (вспомагательный) массив элементов

            switch (nm)
            {
                case NetworkMode.Train:
                    tmpArrStr = File.ReadAllLines(path + "train.txt"); //считывание из файлов обучающей выборки
                    trainset = new double[tmpArrStr.Length, 16]; //определение массива обучающей выборки

                    for (int i = 0; i < tmpArrStr.Length; i++) //цикл перебора строк обучающей выборки
                    {
                        tmpStr = tmpArrStr[i].Split(' '); //разбиение i-й строки на массив отдельных символов 
                        for (int j=0; j<16; j++) //цикл заполнения i-й строки обучающей выборки
                        {
                            trainset[i,j]=double.Parse(tmpStr[j]); //строковое значение преобразуется в 
                        }
                    }
                    Shuffling_Array_Rows(trainset); //перетасовка обучающей выборки методом Фишера-Йетса
                    break;

                case NetworkMode.Test:
                    tmpArrStr = File.ReadAllLines(path + "test.txt"); //считывание из файлов тестовой выборки
                    testset = new double[tmpArrStr.Length, 16]; //определение массива тестовой выборки

                    for (int i = 0; i < tmpArrStr.Length; i++) //цикл перебора строк тестовой выборки
                    {
                        tmpStr = tmpArrStr[i].Split(' '); //разбиение i-й строки на массив отдельных символов 
                        for (int j = 0; j < 16; j++) //цикл заполнения i-й строки тестовой выборки
                        {
                            testset[i, j] = double.Parse(tmpStr[j]); //строковое значение преобразуется в 
                        }
                    }
                    Shuffling_Array_Rows(testset); //перетасовка тествой выборки методом Фишера-Йетса
                    break;
            }
        }

        //перестановка строк массива методом Фишера-Йетса. Fisher-Yates method
        public void Shuffling_Array_Rows(double[,] arr)
        {
            if (arr == null) return;

            Random rand = new Random();
            int rowCount = arr.GetLength(0);
            int colCount = arr.GetLength(1);

            for (int i = rowCount - 1; i > 0; i--)
            {
                // Генерируем случайный индекс от 0 до i
                int j = rand.Next(i + 1);

                // Меняем местами строки i и j
                for (int k = 0; k < colCount; k++)
                {
                    double temp = arr[i, k];
                    arr[i, k] = arr[j, k];
                    arr[j, k] = temp;
                }
            }
        }
    }
}
