#ifndef MY_GLOBALS_H
#define MY_GLOBALS_H

#include <string.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <filesystem>

using namespace std;

//Variáveis
extern int larguraIdeal;
extern int alturaIdeal;
extern float confiancaMinimaDeteccao;
extern int qtdMinimaCadastrada;
extern int qtdSuperiorCadastrada;



//Modelo SSD
extern std::string arquivo_modelo;
extern std::string arquivo_prototxt;
extern cv::dnn::Net network;


//Pastas
extern std::string modelosYmlPath;
extern std::string imagensTeste;
extern std::string arquivos;

//Arquivos
extern std::string caminhosPessoas;
extern std::string todasAsPessoas;
extern std::string pessoasCadastradasFile;
extern std::string normalizadorFile;
extern std::string minValuesFile;
extern std::string maxValuesFile;
extern std::string pessoasKmeans;
extern std::string logs;
#endif
