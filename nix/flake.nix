{
  inputs.nixpkgs.url = "github:nixOS/nixpkgs/nixos-unstable";

  inputs.systems.url = "github:nix-systems/default";
  inputs.parts.url = "github:hercules-ci/flake-parts";
  inputs.parts.inputs.nixpkgs-lib.follows = "nixpkgs";

  outputs = inputs: inputs.parts.lib.mkFlake { inherit inputs; } {
    systems = import inputs.systems;

    perSystem = { lib, pkgs, system, ... }:
      let
        python = pkgs.${pythonVersion};
        pythonVersion = "python313";
      in
        {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              ${pythonVersion} = prev.${pythonVersion}.override {
                packageOverrides = pyFinal: pyPrev:
                  { 
                    ray = pyPrev.ray.overridePythonAttrs ( old: rec {
                      version = "2.47.1";
                      src = let 
                        platforms = {
                          aarch64-darwin = "macosx_12_0_arm64";
                          aarch64-linux = "manylinux2014_aarch64";
                          x86_64-linux = "manylinux2014_x86_64";
                        };
                        pyShortVersion = "cp${builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion}";
                        hashes = {
                          x86_64-linux = "sha256-JSpHHor7kYsQXNv/tMvrsBQ7qtdaBsj/zeJ6wxdXnMs=";
                          aarch64-linux = "";
                          aarch64-darwin = "";
                        };
                      in 

                        pyPrev.fetchPypi {
                          inherit (old) pname format;
                          inherit version;
                          dist = pyShortVersion;
                          python = pyShortVersion;
                          abi = pyShortVersion;
                          platform = platforms.${prev.stdenv.hostPlatform.system} or { };
                          sha256 = hashes.${prev.stdenv.hostPlatform.system} or { };
                        };
                    });

                    dask = pyPrev.dask.overridePythonAttrs ( old: rec {
                      version = "2024.6.0";
                      src = prev.fetchFromGitHub {
                        owner = "dask";
                        repo = "dask";
                        tag = version;
                        hash = "sha256-HtWxVWMk0G2OeBnZKLF5tuOohPbg20Ufl+VH/MX8vK0=";
                      };
                      postPatch = ''
                        # versioneer hack to set version of GitHub package
                        echo "def get_versions(): return {'dirty': False, 'error': None, 'full-revisionid': None, 'version': '${version}'}" > dask/_version.py
                      
                        substituteInPlace setup.py \
                        --replace-fail "import versioneer" "" \
                        --replace-fail "version=versioneer.get_version()," "version='${version}'," \
                        --replace-fail "cmdclass=versioneer.get_cmdclass()," ""
                      
                        substituteInPlace pyproject.toml \
                        --replace-fail ', "versioneer[toml]==0.29"' ""
                        '';
                      dependencies = old.dependencies ++ [pyPrev.numpy pyPrev.pandas];
                      doCheck=false;
                    });

                    doreisa = pyPrev.buildPythonPackage rec{
                      pname = "doreisa";
                      version = "0.3.4";
                      pyproject = true;
                      src = prev.fetchFromGitHub {
                        owner = "deisa-project";
                        repo = "doreisa";
                        tag = "v${version}";
                        hash = "sha256-+W62cinBU//RZZcSuuvL6TCjdRZtGd0Hizy7GeUXH0w=";
                      };
                      build-system = [pyPrev.hatchling];
                      pythonRelaxDeps = ["numpy"];
                      dependencies = with pyFinal; [numpy ray dask ] ++ pyFinal.dask.optional-dependencies.dataframe;
                      doCheck=false;
                    };
                  };
              };
            })
          ];
        };

        devShells.default = pkgs.mkShell {
          USE_NIX=1;
          packages = with pkgs; [
            mpi
            libyaml
            gcc
            imagemagick
            uv
            (python.withPackages (_: with _; [
              ray
              dask
              doreisa

              cloudpickle
              filelock
              google-api-core
              imageio
              ipywidgets
              jsonschema
              matplotlib
              mpi4py
              msgpack
              pillow
              pyyaml
              toolz
              tqdm
            ]
            ))
          ];

          env.LD_LIBRARY_PATH = lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.zlib
            pkgs.mpi
          ];

          shellHook = ''
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            '';
        };

        formatter = pkgs.writeShellScriptBin "formatter" ''
          ${lib.getExe pkgs.nixpkgs-fmt} .
          '';
      };
  };
}
