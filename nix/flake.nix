{
  inputs.nixpkgs.url = "github:nixOS/nixpkgs/nixos-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.parts.url = "github:hercules-ci/flake-parts";
  inputs.parts.inputs.nixpkgs-lib.follows = "nixpkgs";

  inputs.nixpkgs-dask.url = "github:nixOS/nixpkgs/a42717b875f8d9da831ed7a8ceab1dc986ce518b"; # 2024.6.0
  inputs.nixpkgs-ray.url = "github:nixOS/nixpkgs/4f6c942f7a68ee2c05832531d42370c6268e7ff1"; # 2.47.1

  outputs = inputs: inputs.parts.lib.mkFlake { inherit inputs; } {
    systems = import inputs.systems;

    perSystem = { lib, pkgs, system, ... }:
      let
        python = pkgs.${pythonVersion};
        pythonVersion = "python312";
        pythonPkgFrom = nixpkgs: name: {
          ${name} = nixpkgs.legacyPackages.${system}."${pythonVersion}Packages".${name};
        };
      in
      {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              ${pythonVersion} = prev.${pythonVersion}.override {
                packageOverrides = pyFinal: pyPrev:
                  pythonPkgFrom inputs.nixpkgs-dask "dask"
                  //
                  pythonPkgFrom inputs.nixpkgs-ray "ray";
              };
            })
          ];
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            mpi
            libyaml
            gcc
            imagemagick
            (python.withPackages (_: with _; [
              (toPythonModule ray)
              (toPythonModule dask)

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
