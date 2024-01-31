#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.h"


//Variáveis globais
int tempoDeCadastro = 10;
int hiatoDeClusterizacao = 20;
int opcao = 0;
std::string nome = "";
bool continuaGravacao = true;
int fps = 60;
int larguraIdeal = 130;
int alturaIdeal = 180;
float confiancaMinimaDeteccao = 0.9f;
float confiancaMinimaDaPrevisao = 0.8;
int qtdDeDeteccoesPorCluster = 15;
int distanciaMinima = 40;



//Pastas
std::string modeloSSDPath = "modelo_ssd";
std::string pessoasCadastradasPath = "pessoas_cadastradas";
std::string facesPath = "faces";
std::string modelosYmlPath = "modelos";

//Modelos
std::string arquivo_modelo = "modelo_ssd\\res10_300x300_ssd_iter_140000.caffemodel";
std::string arquivo_prototxt = "modelo_ssd\\deploy.prototxt.txt";
cv::dnn::Net network = cv::dnn::readNetFromCaffe(arquivo_prototxt, arquivo_modelo);

//Caminhos
std::string arquivoGravacao = "pessoas_cadastradas\\pessoas.txt";
std::string arquivoId = "pessoas_cadastradas\\ultimoId.txt";
std::string modeloYml = modelosYmlPath + "\\" + "modelo_LBPH.yml";