#include "global.h"

//Variáveis
int larguraIdeal = 130;
int alturaIdeal = 180;
float confiancaMinimaDeteccao = 0.9f;
int qtdMinimaCadastrada = 20;
int qtdSuperiorCadastrada = 150;



//Modelo SSD
std::string arquivo_modelo = "modelo_ssd\\res10_300x300_ssd_iter_140000.caffemodel";
std::string arquivo_prototxt = "modelo_ssd\\deploy.prototxt.txt";
cv::dnn::Net network = cv::dnn::readNetFromCaffe(arquivo_prototxt, arquivo_modelo);


//Pastas
std::string modelosYmlPath = "modelos";
std::string imagensTeste = "ImagensTeste";
std::string arquivos = "Arquivos";

//Arquivos
std::string caminhosPessoas = arquivos + "\\modelosNovos.txt";
std::string todasAsPessoas = arquivos + "\\todasAsPessoas.txt";
std::string pessoasCadastradasFile = arquivos + "\\pessoasCadastradas.txt";
std::string normalizadorFile = arquivos + "\\normalizador.txt";
std::string minValuesFile = arquivos+"\\minValues.txt";
std::string maxValuesFile = arquivos+"\\maxValues.txt";
std::string pessoasKmeans = arquivos + "\\pessoasKmeans.txt";
std::string logs = arquivos + "\\logs.txt";
