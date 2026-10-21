#include "edge/video/engine_artifact.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
int main() {
    namespace fs=std::filesystem;
    const auto dir=fs::temp_directory_path()/("edge-artifact-"+std::to_string(getpid()));
    fs::create_directories(dir);
    try {
        const auto engine=dir/"model.plan";
        std::ofstream(engine)<<"abc";
        const std::map<std::string,std::string> current{{"tensorrt_version","8502"}};
        auto info=edge::video::inspectEngineArtifact(engine,current);
        const std::string hash="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        if(info.at("engine_sha256")!=hash)throw std::runtime_error("SHA256 mismatch");
        auto sidecar=engine.string()+".meta.yaml";
        auto write=[&](const std::string& h,const std::string& version){
            std::ofstream(sidecar)<<"%YAML:1.0\n---\nengine_sha256: \""<<h<<"\"\ntensorrt_version: \""<<version<<"\"\n";
        };
        write(hash,"8502"); edge::video::inspectEngineArtifact(engine,current);
        for(bool badHash:{false,true}) {
            write(badHash?"bad":hash,badHash?"8502":"9999");
            bool rejected=false;
            try { edge::video::inspectEngineArtifact(engine,current); }catch(const std::exception&){rejected=true;}
            if(!rejected)throw std::runtime_error("Mismatching metadata accepted");
        }
        fs::remove_all(dir);std::cout<<"Artifact hash and optional metadata validation passed\n";
    }catch(const std::exception& e){fs::remove_all(dir);std::cerr<<e.what()<<'\n';return 1;}
}
