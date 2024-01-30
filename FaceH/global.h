#ifndef MY_GLOBALS_H
#define MY_GLOBALS_H


#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


//Variáveis globais
extern int tempoDeCadastro;
extern int hiatoDeClusterizacao;
extern int opcao;
extern std::string nome;
extern bool continuaGravacao;
extern int fps;
extern int larguraIdeal;
extern int alturaIdeal;
extern float confiancaMinimaDeteccao;
extern float confiancaMinimaDaPrevisao;
extern int qtdDeDeteccoesPorCluster;
extern int distanciaMinima;



//Pastas
extern std::string modeloSSDPath;
extern std::string pessoasCadastradasPath;
extern std::string facesPath;
extern std::string modelosYmlPath;


//Modelos
extern std::string arquivo_modelo;
extern std::string arquivo_prototxt;
extern cv::dnn::Net network;


//Caminhos
extern std::string arquivoGravacao;
extern std::string arquivoId;
extern std::string modeloYml;



#endif