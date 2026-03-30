{
  description = "Nix flake for ema-pytorch (upstream: lucidrains/ema-pytorch)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
    in {
      packages.${system}.default = pkgs.python3Packages.buildPythonPackage {
        pname = "ema-pytorch";
        version = "setup.py:0.7.9";
        pyproject = true;
        src = ./.;
        build-system = [ pkgs.python3Packages.setuptools ];
        dependencies = with pkgs.python3Packages; [ torch ];
        pythonRelaxDeps = true;
        doCheck = false;
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [ (pkgs.python3.withPackages (ps: with ps; [ torch ])) ];
      };
    };
}
