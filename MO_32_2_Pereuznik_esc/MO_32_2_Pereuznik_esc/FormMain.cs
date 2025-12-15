using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using MO_32_2_Pereuznik_esc.NeuroNet;


namespace MO_32_2_Pereuznik_esc
{
    public partial class FormMain : Form
    {
        private double[] inputPixels; //массив входных данных
        private Network network; //Объявление нейросети

        //Конструктор
        public FormMain()
        {
            InitializeComponent();

            inputPixels = new double[15];
            network = new Network();
        }

        //обработчик события клика кнопки-пикселя
        private void Changing_State_Pixel_Button_Click(object sender, EventArgs e)
        {
            //если кнопка белая
            if (((Button)sender).BackColor == Color.White)
            {
                ((Button)sender).BackColor = Color.Black; //изменение цвета кнопки
                inputPixels[((Button)sender).TabIndex] = 1d; //изменение в массиве
            }
            else
            {
                //если кнопка черная
                ((Button)sender).BackColor = Color.White; //изменение цвета кнопки
                inputPixels[((Button)sender).TabIndex] = 0d; //изменение в массиве
            }
        }

        //сохранить в файл обучающий пример
        private void button_SaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n"; //переход на новую строку
            File.AppendAllText(path, tmpStr); //добавление текста tmpStr
        }                                    // в файл, расположенный по path

        private void button_SaveTestSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "test.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";
            File.AppendAllText(path, tmpStr);
        }


        private void label3_Click(object sender, EventArgs e)
        {

        }

        private void buttonRecognize_Click(object sender, EventArgs e)
        {
            network.ForwardPass(network, inputPixels);
            labelOut.Text=network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            Veroyat.Text = (100 * network.Fact.Max()).ToString("0.00") + " %";
        }

        private void button_training_Click(object sender, EventArgs e)
        {
            network.Train(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]); // ошибка
            }

            for (int i = 0; i < network.Accuracy.Length; i++)
            {
                chart1.Series[0].Points.AddY(network.Accuracy[i]); // точность
            }

            //MessageBox.Show("Обучение успешно завершено. ", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void button_tasting_Click(object sender, EventArgs e)
        {
            network.Test(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]); // ошибка тестирования
            }

            for (int i = 0; i < network.Accuracy.Length; i++)
            {
                chart1.Series[0].Points.AddY(network.Accuracy[i]); // точность тестирования
            }

            //MessageBox.Show("Тестирование успешно завершено. ", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void chart_Eavr_Click(object sender, EventArgs e)
        {

        }

        private void chart1_Click(object sender, EventArgs e)
        {

        }

        private void drop_out_button_Click(object sender, EventArgs e)
        {
            network.DropOutTrain(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]); // ошибка
            }

            for (int i = 0; i < network.Accuracy.Length; i++)
            {
                chart1.Series[0].Points.AddY(network.Accuracy[i]); // точность
            }
            //MessageBox.Show("Dropout применен", "Информация");
        }
    }
}

