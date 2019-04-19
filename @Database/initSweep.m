function obj = initSweep(obj)
    width = obj.ocams{1}.width;
    height = obj.ocams{1}.height;
    if ~isfield(obj.ocams{1}, 'invisible_mask')
        [xs, ys] = meshgrid(1:width, 1:height);
        p = [xs(:) ys(:)]';
        for i = 1:4
            P = cam2world(p, obj.ocams{i});
            p2 = world2cam_fast(P, obj.ocams{i}, obj.opts.max_fov);
            obj.ocams{i}.invisible_mask = reshape(isnan(p2(1,:)), [height width]);
        end
        ocams = obj.ocams;
        poses = obj.poses;
        
        save(fullfile(obj.db_path, 'intrinsic_extrinsic.mat'), 'ocams', 'poses');
    end

    w = obj.sweep.imsize(2);
    h = obj.sweep.imsize(1);
    [xs, ys] = meshgrid(1:w,1:h);
    xs = (xs-w/2)/(w/2)*pi + pi/2;
    if size(obj.sweep.phi, 2) == 1 
        ys = (ys-h/2)/(h/2)*deg2rad(obj.sweep.phi);
    else
        mphi = mean(obj.sweep.phi);
        mphi2 = (obj.sweep.phi(2) - obj.sweep.phi(1)) / 2;
        ys = (ys-h/2)/(h/2)*deg2rad(mphi2) - deg2rad(mphi);
    end
    
    X = -cos(ys).*cos(xs);
    Y = sin(ys); 
    Z = cos(ys).*sin(xs);

    obj.sweep.rays = single([X(:) Y(:) Z(:)]');
    obj.sweep.m_d = 1/obj.sweep.max_depth;
    obj.sweep.M_d = 1/obj.sweep.min_depth;
    obj.sweep.s_d = (obj.sweep.M_d-obj.sweep.m_d)/(obj.sweep.ndisps-1);
    obj.sweep.disps = single(obj.sweep.m_d:obj.sweep.s_d:obj.sweep.M_d);
    
    if obj.sweep.load_grid
        lt_fmt = sprintf('lt_(%d,%d,%d).hwd', h,w, obj.sweep.ndisps);
        lt_file_path = fullfile(obj.db_path, lt_fmt);
        if exist(lt_file_path, 'file')
            fprintf('# Load lookup table: "%s"\n', lt_fmt);
            grids = cell(1,4);
            tmp = readBin(lt_file_path, 'single', [2 h w obj.sweep.ndisps 4]);
            for i = 1:4
               grids{i} = tmp(:,:,:,:,i); 
            end
            obj.sweep.grids = grids;
        else
            fprintf('# Make lookup table...\n');
            obj.sweep.grids = cell(1,4);
            for i = 1:4
                obj.sweep.grids{i} = zeros([2, h, w, obj.sweep.ndisps], 'single');
            end
            for d = 1:obj.sweep.ndisps
                pts = 1/obj.sweep.disps(d)*obj.sweep.rays + ...
                    repmat(obj.sweep.center,1,length(h*w));
                for i = 1:4
                    P = rigidMotion(pts, invext(obj.poses(:,i)));
                    p = world2cam_fast(P, obj.ocams{i}, obj.opts.max_fov);
                    invisible = reshape(isnan(p(1,:)), [h w]);
                    p(:,invisible) = -1e5;
                    p(1,:) = (p(1,:)-1)/(width-1)*2 -1;
                    p(2,:) = (p(2,:)-1)/(height-1)*2 - 1;
                    p = [p(2,:); p(1,:)];
                    obj.sweep.grids{i}(:,:,:,d) = reshape(p, [2 h w]);
                end
            end
            grids = obj.sweep.grids;
            grids = cat(5, grids{1}, grids{2}, grids{3}, grids{4});
            f = fopen(lt_file_path, 'wb');
            fwrite(f, grids, 'single');
            fclose(f);
            fprintf('# Lookup table saved: "%s"\n', lt_file_path);
        end
    end
end

