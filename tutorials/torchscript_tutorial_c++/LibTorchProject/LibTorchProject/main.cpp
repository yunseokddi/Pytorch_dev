#include <torch/script.h> // �ʿ��� �� �ϳ��� �������.

#include <iostream>
#include <memory>

using namespace std;
using namespace torch;

int main(int argc, const char* argv[]) {
	if (argc != 2) {
		cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}


	jit::script::Module module;
	try {
		// torch::jit::load()�� ����� ScriptModule�� ���Ϸκ��� ������ȭ
		module = jit::load(argv[1]);

	}
	catch (const c10::Error& e) {
		cerr << "error loading the model\n";
		return -1;
	}

	cout << "ok\n";

	vector<jit::IValue> inputs;
	inputs.push_back(torch::ones({ 1,3,224,224 }));

	at::Tensor output = module.forward(inputs).toTensor();
	cout << output.slice(1, 0, 5) << endl;
}