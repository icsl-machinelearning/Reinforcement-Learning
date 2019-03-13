#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define GENMAX 1000
#define NODENO 15
#define ALPHA 0.1
#define GAMMA 0.9
#define EPSILON 0.3
#define SEED 32769

int rand100();
int rand01();
double rand1();
void printqvalue(int *qvalue);
int selecta(int s, int *qvalue);
int updateq(int s, int *qvalue);

int main() {
	int i;
	int	s; //����
	int t;  // �ð�
	int qvalue[NODENO]; //q��
	srand((unsigned int)time(NULL));

	/*Q�� �ʱ�ȭ*/
	for (i = 0; i < NODENO; i++)
		qvalue[i] = rand100();
	printqvalue(qvalue);

	/*���� �н�*/
	for (i = 0; i < GENMAX; i++) {
		s = 0;
		for (t = 0; t < 3; t++) {
			s = selecta(s, qvalue);
			qvalue[s] = updateq(s, qvalue);
		}
		printqvalue(qvalue);
	}
	return 0;
}

int updateq(int s, int *qvalue) {
	int qv;
	int qmax;

	if (s > 6) {
		if (s == 14)
			qv = qvalue[s] + ALPHA*(1000 - qvalue[s]);
		else if (s == 11)
			qv = qvalue[s] + ALPHA*(500 - qvalue[s]);
		else
			qv = qvalue[s];
	}
	else {
		if ((qvalue[2 * s + 1]) > (qvalue[2*s + 2]))
			qmax = qvalue[2 * s + 1];
		else
			qmax = qvalue[2 * s + 2];
		qv = qvalue[s] + ALPHA*(GAMMA*qmax - qvalue[s]);
	}
	return qv;
}

int selecta(int olds, int *qvalue) {
	int s;
	//double e = 1.0 / ((i/100)+1);
	if (rand1() < EPSILON ){
		if (rand01() == 0)
			s = 2 * olds + 1;
		else
			s = 2 * olds + 2;
	}
	else {
		if ((qvalue[2 * olds + 1]) > (qvalue[2 * olds + 2]))
			s = 2 * olds + 1;
		else
			s = 2 * olds + 2;
	}
	return s;
}

void printqvalue(int *qvalue) {
	int i;
	
	for (i = 1; i<NODENO; i++)
		printf("%d\t", qvalue[i]);
	printf("\n");
}

double rand1() {
	return (double)rand() / RAND_MAX;
}

int rand01() {
	int rnd;
	
	while ((rnd = rand()) == RAND_MAX);
	return (int)((double)rnd / RAND_MAX * 2);
}

int rand100() {
	int rnd;

	while ((rnd = rand()) == RAND_MAX);
	return (int)((double)rnd / RAND_MAX * 101);
}