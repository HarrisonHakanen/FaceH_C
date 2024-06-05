#include <string.h>

#include "global.h"
#include "global_kmeans.h"

int quantDimensoes = 70;
int valorMaximo = 1;
int qtdCentroides = 10;
std::string arquivoCentroides = arquivos + "\\centroides.txt";
std::string arquivoCentroidesAux = arquivos + "\\centroidesAux.txt";
std::string arquivo = "Arquivos";
int qtdMinimaCadastradaRemodelarCentroides = 100;
int qtdMinimaPorCluster = 80;