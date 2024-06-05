#ifndef FUNCTIONSKMEANS_H
#define FUNCTIONSKMEANS_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <math.h>
#include <filesystem>


struct Centroid{
	int id;
	std::vector<float> dimensoes;
	float distancia;
};


struct Registro {
	int id;
	std::vector<float> dimensoes;
	Centroid classe;
	std::string caminhoDoArquivo;
	std::string nome;
};


struct RegistroNew {
	int id;
	std::vector<std::vector<float>> dimensoes;
	Centroid classe;
	std::string caminhoDoArquivo;
	std::string nome;
};


struct RetornoCentroides {
	std::vector<Centroid> centroides;
	bool alterou;
};


Centroid calculoDistancia(std::vector<float> dimensoes, Centroid centroid);
std::vector<Registro> classificaRegistros(std::vector<Registro> registros, std::vector<Centroid> centroides);
Centroid classificaRegistro(std::vector<float> registros, std::vector<Centroid> centroides);
std::vector<Registro> separaOsClusters(std::vector<Registro> registros, int indexReg);
RetornoCentroides reposicionaClusters(std::vector<Registro> registrosCluster, std::vector<Centroid>centroides);
std::vector<float>geraDimensoes();
void escreverArquivoKmeans(std::string nomeArquivo, auto conteudo);
void criarCentroides(std::string arquivoCentroides);
std::vector<float>geraDimensoes();
std::vector<float>geraDimensoesComIntervalo(int rangeMax, int rangeMin);
#endif
