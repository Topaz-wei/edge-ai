#include "edge/video/engine_artifact.hpp"
#include <opencv2/core.hpp>
#include <openssl/evp.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
namespace edge::video {
std::map<std::string,std::string> inspectEngineArtifact(
    const std::filesystem::path& engine,
    const std::map<std::string,std::string>& current) {
    std::ifstream file(engine,std::ios::binary);
    if(!file) throw std::runtime_error("Cannot read engine: "+engine.string());
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if(!digest || EVP_DigestInit_ex(digest.get(),EVP_sha256(),nullptr)!=1)
        throw std::runtime_error("Cannot initialize SHA256");
    std::array<char,65536> buffer{};
    while(file) {
        file.read(buffer.data(),buffer.size());
        if(file.gcount() && EVP_DigestUpdate(digest.get(),buffer.data(),file.gcount())!=1)
            throw std::runtime_error("Cannot update SHA256");
    }
    if(!file.eof()) throw std::runtime_error("Cannot finish reading engine");
    std::array<unsigned char,EVP_MAX_MD_SIZE> bytes{}; unsigned length=0;
    if(EVP_DigestFinal_ex(digest.get(),bytes.data(),&length)!=1)
        throw std::runtime_error("Cannot finalize SHA256");
    std::ostringstream hash;
    for(unsigned i=0;i<length;++i) hash<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(bytes[i]);
    auto values=current; values["engine_sha256"]=hash.str();
    const auto sidecar=engine.string()+".meta.yaml";
    if(std::filesystem::exists(sidecar) || std::filesystem::is_symlink(std::filesystem::symlink_status(sidecar))) {
        cv::FileStorage metadata(sidecar,cv::FileStorage::READ);
        if(!metadata.isOpened() || !metadata.root().isMap() || metadata["engine_sha256"].empty())
            throw std::runtime_error("Engine metadata must include engine_sha256: "+sidecar);
        std::set<std::string> seen;
        for(auto i=metadata.root().begin();i!=metadata.root().end();++i) {
            auto node=*i; const auto key=node.name(); const auto actual=values.find(key);
            if(!seen.insert(key).second || actual==values.end() || !node.isString() || std::string(node)!=actual->second)
                throw std::runtime_error("Engine metadata mismatch or unsupported key: "+key);
        }
    }
    return values;
}
}
