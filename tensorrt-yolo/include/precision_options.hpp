#pragma once
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace precision {
struct Options {
    std::string model, engine, mode, calibration, cache, timing, layers, dump;
    bool inspect=false;
};
inline std::vector<std::string> readNames(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read layer list: " + path);
    std::vector<std::string> result;
    std::set<std::string> seen;
    std::string line;
    while (std::getline(file,line)) {
        auto first=line.find_first_not_of(" \t\r");
        if(first==std::string::npos || line[first]=='#') continue;
        auto name=line.substr(first,line.find_last_not_of(" \t\r")-first+1);
        if(!seen.insert(name).second) throw std::runtime_error("Duplicate layer: " + name);
        result.push_back(name);
    }
    if(file.bad()) throw std::runtime_error("Layer list read error: " + path);
    return result;
}
inline Options parse(const std::vector<std::string>& args) {
    if(args.size()<3) throw std::runtime_error("Expected model, engine and precision");
    Options o; o.model=args[0]; o.engine=args[1]; o.mode=args[2];
    if(o.mode!="fp32" && o.mode!="fp16" && o.mode!="int8") throw std::runtime_error("Invalid precision");
    std::vector<std::string> pos; std::set<std::string> seen;
    for(std::size_t i=3;i<args.size();++i) {
        auto key=args[i];
        if(key.rfind("--",0)!=0) {pos.push_back(key);continue;}
        if(!seen.insert(key).second) throw std::runtime_error("Duplicate option: "+key);
        if(key=="--inspect-only") {o.inspect=true;continue;}
        std::string* target=nullptr;
        if(key=="--timing-cache") target=&o.timing;
        else if(key=="--fp16-layers") target=&o.layers;
        else if(key=="--dump-layers") target=&o.dump;
        else throw std::runtime_error("Unknown option: "+key);
        if(++i>=args.size() || args[i].empty() || args[i].rfind("--",0)==0) throw std::runtime_error("Missing value: "+key);
        *target=args[i];
    }
    if(o.mode=="int8") {
        if(pos.size()>2 || (pos.empty() && !o.inspect)) throw std::runtime_error("INT8 requires calibration directory and optional cache");
        if(!pos.empty()) o.calibration=pos[0];
        o.cache=pos.size()==2?pos[1]:o.engine+".cache";
    } else if(!pos.empty() || !o.layers.empty()) throw std::runtime_error("Calibration/layer list require INT8");
    if(o.inspect && o.dump.empty()) throw std::runtime_error("--inspect-only requires --dump-layers");
    std::vector<std::string> files{o.model,o.engine,o.cache,o.timing,o.layers,o.dump};
    std::vector<std::string> temporary{o.engine+".tmp"};
    if(!o.cache.empty()) {
        files.push_back(o.cache+".meta");
        temporary.push_back(o.cache+".tmp");temporary.push_back(o.cache+".meta.tmp");
    }
    if(!o.timing.empty()) temporary.push_back(o.timing+".tmp");
    files.insert(files.end(),temporary.begin(),temporary.end());
    std::vector<std::filesystem::path> paths;
    for(const auto& p:files) {
        if(p.empty())continue;
        auto normalized=std::filesystem::weakly_canonical(p);
        for(const auto& previous:paths) {
            if(normalized==previous || (std::filesystem::exists(normalized) && std::filesystem::exists(previous)
                && std::filesystem::equivalent(normalized,previous)))
                throw std::runtime_error("Input/output/cache/temporary paths must be distinct: "+p);
        }
        paths.push_back(normalized);
    }
    for(const auto& p:temporary)
        if(std::filesystem::exists(p) || std::filesystem::is_symlink(std::filesystem::symlink_status(p)))
            throw std::runtime_error("Temporary path already exists: "+p);
    return o;
}
}
