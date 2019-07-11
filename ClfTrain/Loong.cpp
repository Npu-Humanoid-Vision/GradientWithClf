#include <Params.h>
#include <Util.h>

int main(int argc, char const *argv[]) {
    string m_p = Train_csvc(0.0);
    Evaluation(m_p);

    CvSVM tester;
    tester.load(m_p.c_str());
    
    return 0;
}
