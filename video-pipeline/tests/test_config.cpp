#include "edge/video/config.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
namespace fs=std::filesystem;
int main() {
    const auto dir=fs::temp_directory_path()/("edge-config-"+std::to_string(getpid()));
    fs::create_directories(dir);
    try {
        const auto path=dir/"run.yaml";
        auto write=[&](const std::string& text){std::ofstream(path)<<"%YAML:1.0\n---\n"<<text;};
        const std::string valid="schema_version: 1\nsource: \"clip.mp4\"\ndisplay: 0\nmax_frames: 3\nmodels:\n  - { id: person, type: yolo, engine: \"model.plan\", input_stage: business, graph: 1 }\n";
        write(valid);
        const auto c=edge::video::loadConfig(path);
        if(c.models.size()!=1 || c.models[0].engine!=dir/"model.plan" || c.source!=(dir/"clip.mp4").string() || c.display || c.maxFrames!=3) throw std::runtime_error("Config parsing/path mismatch");
        auto old=fs::current_path(); fs::current_path("/tmp");
        const auto other=edge::video::loadConfig(path); fs::current_path(old);
        if(other.models[0].engine!=c.models[0].engine) throw std::runtime_error("cwd changes path");
        auto rejects=[&](const std::string& text){write(text); bool bad=false;try {edge::video::loadConfig(path);}catch(const std::exception&){bad=true;}if(!bad)throw std::runtime_error("Accepted invalid config: "+text);};
        for (const std::string value : {"4294967297", "-4294967295", "0x100000001"}) {
            auto overflow = valid;
            overflow.replace(overflow.find("max_frames: 3"), 13, "max_frames: " + value);
            rejects(overflow);
        }
        for (const std::string field : {"schema_version: 1", "display: 0", "graph: 1"}) {
            auto overflow=valid;
            const auto start=overflow.find(field);
            overflow.replace(start,field.size(),field.substr(0,field.find(':')+1)+" 4294967297");
            rejects(overflow);
        }
        auto quoted=valid;
        quoted.replace(quoted.find("clip.mp4"),8,"4294967297.mp4");
        write(quoted);
        if(edge::video::loadConfig(path).source!=(dir/"4294967297.mp4").string())
            throw std::runtime_error("Quoted path incorrectly treated as integer");
        auto apostrophe=valid;
        apostrophe.replace(apostrophe.find("\"clip.mp4\""),10,"clip's.mp4");
        apostrophe.replace(apostrophe.find("max_frames: 3"),13,"max_frames: 4294967297");
        rejects(apostrophe);
        rejects(valid+"typo: 3\n");
        rejects("schema_version: 2\nsource: x\nmodels: []\n");
        rejects("schema_version: 1\nsource: x\nmodels: []\n");
        rejects(valid+"capture_width: 640\n");
        rejects(valid+"max_frames: 0\n");
        rejects(valid+"display: \"false\"\n");
        rejects(valid+"  - { id: person, type: yolo, engine: x, input_stage: business }\n");
        auto bad=valid; bad.replace(bad.find("graph: 1"),8,"graph: 2"); rejects(bad);
        write("schema_version: 1\nsource: \"0\"\nmodels:\n  - { id: person, type: yolo, engine: x, input_stage: camera }\n");
        if(!edge::video::loadConfig(path).camera) throw std::runtime_error("Camera not recognized");
        fs::remove_all(dir); std::cout<<"Strict config and path tests passed\n";
    } catch(const std::exception& e) { fs::remove_all(dir); std::cerr<<e.what()<<'\n'; return 1; }
}
